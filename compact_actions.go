// Copyright 2026 KGV FusionX
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package blueprint

import (
	"bufio"
	"bytes"
	"compress/flate"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// Compact build actions: low-memory hosts can set BLUEPRINT_COMPACT_ACTIONS=1
// to have each module's build actions rendered to ninja text and compressed
// as soon as the module has generated them, instead of being held as parsed
// buildDefs until the whole build file is written. The parsed form is most of
// soong_build's peak heap on a full platform build.
//
// Rendering early needs the package names before liveness is known, so in
// this mode names are assigned over every registered package. That can only
// lengthen a name (when a short name collides with a package that ends up
// unused), which still produces a valid, self-consistent build file.
//
// Anything that reads module.actionDefs.buildDefs after PrepareBuildActions
// (AllTargets, the JSON action graph) sees no module actions in this mode.

var compactActionsEnabled = os.Getenv("BLUEPRINT_COMPACT_ACTIONS") != ""

var flateWriters = sync.Pool{
	New: func() interface{} {
		w, err := flate.NewWriter(nil, flate.BestSpeed)
		if err != nil {
			panic(err)
		}
		return w
	},
}

type compactActions struct {
	data              []byte
	endsWithBlankLine bool
}

// allPackageNames gives every registered package a unique name, using the
// same short/long rules as makeUniquePackageNames.
func allPackageNames() map[*packageContext]string {
	pkgs := make(map[string]*packageContext)
	pkgNames := make(map[*packageContext]string)
	longPkgNames := make(map[*packageContext]bool)
	for _, pctx := range packageContexts {
		if other, present := pkgs[pctx.shortName]; present {
			longPkgNames[pctx] = true
			longPkgNames[other] = true
		} else {
			pkgNames[pctx] = pctx.shortName
			pkgs[pctx.shortName] = pctx
		}
	}
	for pctx := range longPkgNames {
		pkgNames[pctx] = pctx.fullName
	}
	return pkgNames
}

func (c *Context) compactPackageNames() map[*packageContext]string {
	c.compactPkgNamesOnce.Do(func() {
		c.compactPkgNames = allPackageNames()
	})
	return c.compactPkgNames
}

// compactModuleActions renders module's local build actions and drops the
// parsed form. It must run after processLocalBuildActions has recorded the
// actions' dependencies in the live tracker.
func (c *Context) compactModuleActions(module *moduleInfo) error {
	defs := &module.actionDefs
	if len(defs.variables)+len(defs.rules)+len(defs.buildDefs) == 0 {
		return nil
	}

	var buf bytes.Buffer
	fw := flateWriters.Get().(*flate.Writer)
	defer flateWriters.Put(fw)
	fw.Reset(&buf)
	bw := bufio.NewWriter(fw)
	nw := newNinjaWriter(bw)

	if err := c.writeLocalBuildActions(nw, defs, c.compactPackageNames()); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	if err := fw.Close(); err != nil {
		return err
	}

	module.compactActions = &compactActions{
		data:              append([]byte(nil), buf.Bytes()...),
		endsWithBlankLine: nw.justDidBlankLine,
	}
	module.actionDefs = localBuildActions{}
	return nil
}

// writeCompactActions writes rendered actions back out in place of
// writeLocalBuildActions.
func writeCompactActions(nw *ninjaWriter, actions *compactActions) error {
	r := flate.NewReader(bytes.NewReader(actions.data))
	defer r.Close()
	chunk := make([]byte, 64<<10)
	for {
		n, err := r.Read(chunk)
		if n > 0 {
			if _, werr := nw.writer.WriteString(string(chunk[:n])); werr != nil {
				return werr
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	nw.justDidBlankLine = actions.endsWithBlankLine
	return nil
}

// Low-memory diagnostics: BLUEPRINT_VARIANT_STATS=<file> records how many
// module variants each module type and variant flavour produced, and how far
// build action generation has got, with the heap size at each step.

var variantStatsFile = os.Getenv("BLUEPRINT_VARIANT_STATS")

func (c *Context) writeVariantStats() {
	if variantStatsFile == "" {
		return
	}
	byType := map[string]int{}
	byToken := map[string]int{}
	for _, m := range c.modulesSorted {
		byType[m.typeName]++
		for _, token := range strings.Split(m.variant.name, "_") {
			byToken[token]++
		}
	}
	f, err := os.Create(variantStatsFile)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "variants %d\n", len(c.modulesSorted))
	for _, counts := range []map[string]int{byType, byToken} {
		keys := make([]string, 0, len(counts))
		for k := range counts {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool { return counts[keys[i]] > counts[keys[j]] })
		for i, k := range keys {
			if i == 60 {
				break
			}
			fmt.Fprintf(f, "%8d %s\n", counts[k], k)
		}
		fmt.Fprintln(f, "--")
	}
}

var generatedModules int64

func noteGeneratedModule() {
	if variantStatsFile == "" {
		return
	}
	if n := atomic.AddInt64(&generatedModules, 1); n%2000 == 0 {
		var stats runtime.MemStats
		runtime.ReadMemStats(&stats)
		f, err := os.OpenFile(variantStatsFile, os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return
		}
		fmt.Fprintf(f, "generated %d heap %dM\n", n, stats.HeapAlloc>>20)
		f.Close()
	}
}

// BLUEPRINT_SKIP_MODULE_TYPES=<type>,<type>... drops every module of those
// types while parsing Android.bp files, before any variants are created. Low-
// memory device builds use it for test and fuzz modules, which no image
// installs; references to them are left to ALLOW_MISSING_DEPENDENCIES.
var skippedModuleTypes = func() map[string]bool {
	types := map[string]bool{}
	for _, t := range strings.Split(os.Getenv("BLUEPRINT_SKIP_MODULE_TYPES"), ",") {
		if t = strings.TrimSpace(t); t != "" {
			types[t] = true
		}
	}
	return types
}()
