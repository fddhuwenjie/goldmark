package parser

import (
	"bytes"
	"errors"
	"fmt"
	"sort"

	"github.com/yuin/goldmark/v2/ast"
	"github.com/yuin/goldmark/v2/text"
	"github.com/yuin/goldmark/v2/util"
)

// ErrInvalidEditRange is returned by [EditableDocument.ApplyEdit] when the
// given edit range is out of bounds of the current source or when start is
// greater than end.
// The document session is left unchanged when this error is returned.
var ErrInvalidEditRange = errors.New("parser: invalid edit range")

// EditableDocument is a Markdown document session that supports applying
// byte-level edits incrementally.
//
// After every successful edit, the document returned by AST and its rendered
// output are equivalent to parsing the updated source from scratch with the
// same [Parser] configuration, while block nodes whose semantics are not
// affected by the edit keep their identity: their block parser callbacks are
// not invoked again.
//
// All byte slices passed to an EditableDocument are copied: later mutations of
// the slices made by the caller never affect the session.
type EditableDocument interface {
	// AST returns the current document tree.
	// The returned tree is owned by the session and must not be modified.
	// It is replaced after every successful edit; callers must not retain
	// nodes across edits except to compare identities.
	AST() ast.Node

	// Source returns a copy of the current Markdown source.
	Source() []byte

	// ApplyEdit replaces the source range [start, end) with replacement.
	//
	// It returns [ErrInvalidEditRange] (possibly wrapped) if start or end is
	// out of bounds or if start > end; in that case the session keeps its
	// previous state.
	//
	// A no-op edit (including start == end with empty replacement, or
	// replacement identical to the replaced range) succeeds without reparsing
	// or invalidating any node.
	ApplyEdit(start, end int, replacement []byte) error
}

type editableDocument struct {
	p      Parser
	mp     *parser
	source []byte
	root   *ast.Document
	pc     *parseContext
}

// NewEditableDocument creates a new editable document session for the given
// source, parsed with the given [Parser].
//
// The source slice is copied and never retained.
func NewEditableDocument(p Parser, source []byte) EditableDocument {
	src := append([]byte(nil), source...)
	d := &editableDocument{
		p:      p,
		source: src,
	}
	if mp, ok := p.(*parser); ok {
		d.mp = mp
		// A context without a fixed generator: Parse installs the parser's
		// ID generator after its lazy initialization has run.
		ctx := newContext(nil, false).(*parseContext)
		ctx.ownedLinkDefs = true
		d.root = p.Parse(src, WithContext(ctx)).(*ast.Document)
		d.pc = ctx
	} else {
		d.root = p.Parse(src).(*ast.Document)
	}
	return d
}

// AST implements [EditableDocument].AST .
func (d *editableDocument) AST() ast.Node {
	return d.root
}

// Source implements [EditableDocument].Source .
func (d *editableDocument) Source() []byte {
	return append([]byte(nil), d.source...)
}

// ApplyEdit implements [EditableDocument].ApplyEdit .
func (d *editableDocument) ApplyEdit(start, end int, replacement []byte) error {
	if start < 0 || end < start || end > len(d.source) {
		return fmt.Errorf("%w: range [%d:%d) is not within source of length %d",
			ErrInvalidEditRange, start, end, len(d.source))
	}
	if bytes.Equal(replacement, d.source[start:end]) {
		// Empty edits and text-identical replacements change nothing: keep
		// every node and do not run any parser callback.
		return nil
	}

	newSource := make([]byte, 0, len(d.source)-end+start+len(replacement))
	newSource = append(newSource, d.source[:start]...)
	newSource = append(newSource, replacement...)
	newSource = append(newSource, d.source[end:]...)

	if d.mp == nil {
		root, ok := d.p.Parse(newSource).(*ast.Document)
		if !ok {
			return fmt.Errorf("parser: unexpected document node type %T", root)
		}
		d.root = root
		d.source = newSource
		return nil
	}

	return d.applyIncremental(start, end, newSource)
}

func (d *editableDocument) newParseContext() *parseContext {
	pc := newContext(d.mp.idGenerator, true).(*parseContext)
	pc.inlineParsers = &d.mp.inlineParsers
	pc.ownedLinkDefs = true
	return pc
}

// copyUsedIDs seeds dst with the supplied heading id names.
func copyUsedIDs(dst *IDs, names []string) {
	if dst.values == nil {
		dst.values = make(map[string]bool, len(names))
	}
	for _, v := range names {
		dst.values[v] = true
	}
}

// applyIncremental reparses only the window of top-level blocks affected by
// the edit, splices the result into the existing tree (shifting source
// positions of reused blocks), refreshes inline content that depends on
// changed link definitions and runs the AST transformers on the whole tree.
func (d *editableDocument) applyIncremental(start, end int, newSource []byte) error {
	oldSource := d.source
	delta := len(newSource) - len(oldSource)
	decoder := d.mp.decoder

	children := childBlocks(d.root)
	oldLines := buildLineStarts(oldSource)
	// Block bounds are aligned to the start of their first source line:
	// window arithmetic works on line ranges, and block Positions may point
	// past leading indentation.
	bounds := make([]int, len(children))
	for i, c := range children {
		bounds[i] = oldLines[lineIndex(oldLines, blockStartOffset(c))]
	}

	// Locate the top-level blocks intersected by the edited line range.
	var lo, hi int
	var wStartOld, wEndOld int
	if len(children) == 0 {
		lo, hi = -1, -1
		wStartOld, wEndOld = 0, len(oldSource)
	} else {
		startLine := oldLines[lineIndex(oldLines, start)]
		lo = sort.Search(len(bounds), func(i int) bool { return bounds[i] > startLine }) - 1
		if lo < 0 {
			lo = 0
		}
		editEndLine := startLine
		if end > start {
			editEndLine = oldLines[lineIndex(oldLines, end-1)]
		}
		hi = sort.Search(len(bounds), func(i int) bool { return bounds[i] > editEndLine }) - 1
		if hi < 0 {
			hi = 0
		}
		if hi < lo {
			hi = lo
		}
		wStartOld = 0
		if lo > 0 {
			wStartOld = bounds[lo]
		}
		wEndOld = len(oldSource)
		if hi+1 < len(bounds) {
			wEndOld = bounds[hi+1]
		}
	}

	// Window coordinates in the new source. Everything before the edit point
	// is unchanged, and the initial window ends on or after the edit point.
	replEnd := start + (end - start) + delta
	wStartNew := wStartOld
	wEndNew := wEndOld + delta

	// Expand across block boundaries that can couple with their neighbors
	// (adjacent lines, setext underlines, open code fences, loose lists and
	// so on).
	newLines := buildLineStarts(newSource)
	listAt := func(newOffset int) *ast.List {
		return d.oldListAt(newOffset, replEnd, delta, bounds, children)
	}
	canCrossBack := func(prevLine, atLine int) bool {
		return d.oldBlockIncludableBackward(newSource, newLines, prevLine, atLine, bounds, children)
	}
	listAcrossBlank := func(prevLine, atLine int) bool {
		return d.oldListContinuesAt(newSource, newLines, prevLine, atLine, bounds, children)
	}
	wStartNew, wEndNew = expandWindow(newSource, newLines, wStartNew, wEndNew, listAt, canCrossBack, listAcrossBlank)

	// A block immediately followed by the reparsed region across a blank line
	// can have its blank-previous-lines flag changed by the edit; widen the
	// window over such blocks (and any blocks their structure couples with)
	// so that their flags, including those of their first descendants, are
	// recomputed by a real parse.
	if len(children) > 0 {
		for {
			i := sort.SearchInts(newLines, wEndNew)
			for i < len(newLines) && util.IsBlank(lineBytes(newSource, newLines, i)) {
				i++
			}
			if i == len(newLines) {
				break
			}
			j := sort.Search(len(bounds), func(k int) bool { return bounds[k] >= newLines[i]-delta })
			if j >= len(children) {
				break
			}
			if children[j].(ast.BlockNode).HasBlankPreviousLines() {
				break
			}
			nextEnd := len(newSource)
			if j+1 < len(bounds) {
				nextEnd = bounds[j+1] + delta
			}
			wStartNew, wEndNew = expandWindow(newSource, newLines, wStartNew, nextEnd, listAt, canCrossBack, listAcrossBlank)
		}
	}

	// Window boundary bookkeeping is finalized by the fence-stability loop
	// below, which may extend the window further.
	wStartOld2 := wStartNew
	wEndOld2 := end
	var rlo, rhi = lo, hi

	// Reparse the window as a standalone document. Positions are relative to
	// the snippet and shifted afterwards, so this is equivalent to parsing
	// the region in place. The block parse is repeated while it produces a
	// fenced code block that runs off the snippet end: such a fence swallows
	// following lines (possibly in nested containers) and the window must be
	// extended up to its closing fence.
	var (
		tmp    *ast.Document
		tpc    *parseContext
		reader text.Reader
	)
	for range 16 {
		snippet := newSource[wStartNew:wEndNew]
		tmp = ast.NewDocument()
		tpc = d.newParseContext()
		// Only heading ids established before the window are in scope for
		// new window headings; ids from replaced blocks must not pre-occupy
		// the table.
		copyUsedIDs(tpc.ids, priorHeadingIDs(d.root, wStartOld2))
		reader = text.NewReader(snippet, decoder)
		d.mp.parseBlocks(tmp, reader, tpc)

		nextEnd := wEndNew
		// An unclosed fence produced by the reparsed window keeps consuming
		// following lines until its closer (or the document end).
		if fenceEnd, open := windowFenceEnd(newSource, buildLineStarts(newSource),
			wStartNew, wEndNew, snippet, tmp); open && fenceEnd > nextEnd {
			nextEnd = fenceEnd
		}
		// The window must span every old block it touches: a block whose
		// start lies before the window end may continue well past it (for
		// example an unclosed fence that loses or gains its closer).
		if len(children) > 0 {
			oldEnd := end
			if wEndNew >= replEnd {
				oldEnd = wEndNew - delta
			}
			j := sort.Search(len(bounds), func(k int) bool { return bounds[k] >= oldEnd }) - 1
			if j >= 0 {
				blockEnd := len(oldSource)
				if j+1 < len(bounds) {
					blockEnd = bounds[j+1]
				}
				if blockEnd+delta > nextEnd {
					nextEnd = blockEnd + delta
				}
			}
		}
		if nextEnd <= wEndNew {
			break
		}
		wEndNew = nextEnd
	}

	// Map the final window back to old-source coordinates (extensions may
	// have grown it beyond the initially computed range).
	if wEndNew >= replEnd {
		wEndOld2 = wEndNew - delta
	}
	rlo, rhi = lo, hi
	if len(children) > 0 {
		rlo = sort.Search(len(bounds), func(i int) bool { return bounds[i] >= wStartOld2 })
		if rlo >= len(bounds) {
			rlo = len(bounds) - 1
		}
		rhi = sort.Search(len(bounds), func(i int) bool { return bounds[i] >= wEndOld2 }) - 1
		if rhi < 0 {
			rhi = 0
		}
		if rhi < rlo {
			rhi = rlo
		}
	}

	// The standalone parse marks its first container chain (list item,
	// blockquote paragraph and so on) as having blank previous lines;
	// recompute the flag from the line actually preceding the window (the
	// blocks may interrupt a non-blank preceding block) and propagate it
	// down the first-child spine.
	if first := tmp.FirstChild(); first != nil {
		blank := true
		if wStartNew > 0 {
			firstLine := lineIndex(newLines, wStartNew)
			switch {
			case firstLine == 0:
				blank = true
			case util.IsBlank(lineBytes(newSource, newLines, firstLine)):
				// The edit blanked the window's first line: the first real
				// block opens only after skipping blanks.
				blank = true
			case util.IsBlank(lineBytes(newSource, newLines, firstLine-1)):
				blank = true
			default:
				// Adjacent non-blank predecessor: blocks normally continue
				// parsing on the same iteration, but a fence closes on its
				// closer line and a definition closes the paragraph phase,
				// so the following block starts a fresh outer iteration.
				blank = false
				if rlo-1 >= 0 {
					switch prev := children[rlo-1].(type) {
					case *ast.CodeBlock:
						blank = prev.CodeBlockKind == ast.CodeBlockKindFenced
					case *ast.LinkReferenceDefinition:
						blank = true
					}
				}
			}
		}
		setLeftSpineBlankFlags(first, blank)
	}

	// Link definitions that live outside the reparsed window must stay
	// visible while its paragraphs are parsed inline, because CommonMark
	// definitions can be referenced from anywhere in the document.
	outsideDefsBefore, outsideDefsAfter := d.outsideLinkDefinitions(wStartOld2, wEndOld2)
	for _, ld := range outsideDefsBefore {
		tpc.AddLinkDefinition(ld)
	}
	for _, ld := range outsideDefsAfter {
		tpc.AddLinkDefinition(ld)
	}
	blockReader := text.NewBlockReader(reader.Source(), nil, decoder)
	d.mp.walkBlock(tmp, func(node ast.Node) {
		d.mp.parseBlock(blockReader, node, tpc)
	})

	// Remember every node produced by the reparse: those nodes must not be
	// treated as reusable blocks during the reference-definition refresh.
	windowNodes := make(map[ast.Node]bool)
	for c := tmp.FirstChild(); c != nil; c = c.NextSibling() {
		collectNodes(c, windowNodes)
	}

	// Splice the reparsed blocks into the existing document.
	for i := rlo; i <= rhi; i++ {
		d.root.RemoveChild(children[i])
	}
	var anchor ast.Node
	if rhi+1 < len(children) {
		anchor = children[rhi+1]
		for i := rhi + 1; i < len(children); i++ {
			shiftTree(children[i], delta)
		}
		// The edit can insert blank lines directly before a reused block;
		// recompute its blank-previous-lines flag from the new source and
		// propagate it down the first-child spine.
		if _, ok := anchor.(ast.BlockNode); ok {
			newLines := buildLineStarts(newSource)
			anchorLine := lineIndex(newLines, blockStartOffset(anchor))
			blank := anchorLine == 0
			if !blank {
				blank = util.IsBlank(lineBytes(newSource, newLines, anchorLine-1))
			}
			// A fence closes on its own closer line and a definition ends
			// the paragraph phase; the next block starts a fresh outer
			// parser iteration in both cases.
			if last := tmp.LastChild(); !blank && last != nil {
				switch v := last.(type) {
				case *ast.CodeBlock:
					blank = v.CodeBlockKind == ast.CodeBlockKindFenced
				case *ast.LinkReferenceDefinition:
					blank = true
				}
			}
			setLeftSpineBlankFlags(anchor, blank)
		}
	}
	for c := tmp.FirstChild(); c != nil; {
		next := c.NextSibling()
		shiftTree(c, wStartNew)
		if anchor != nil {
			d.root.InsertBefore(anchor, c)
		} else {
			d.root.AppendChild(c)
		}
		c = next
	}

	// Build the final parse context: link definitions in document order from
	// the final tree (first definition wins, as in a full parse).
	fpc := d.newParseContext()
	_ = ast.Walk(d.root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if ld, ok := n.(*ast.LinkReferenceDefinition); ok {
			fpc.AddLinkDefinition(newLinkDefinitionFromNode(ld, newSource))
		}
		return ast.WalkContinue, nil
	})
	copyUsedIDs(fpc.ids, priorHeadingIDs(d.root, wStartNew))
	normalizeHeadingIDs(d.root, fpc.ids)

	// When reference definitions changed, blocks that depend on them must be
	// re-parsed inline; blocks without references keep their identity.
	d.refreshDependentBlocks(newSource, fpc, windowNodes)

	// Whole-tree transformers always process the complete new document.
	fullReader := text.NewReader(newSource, decoder)
	for _, at := range d.mp.astTransformers {
		at.Transform(d.root, fullReader, fpc)
	}

	d.source = newSource
	d.pc = fpc
	return nil
}

// outsideLinkDefinitions returns link definitions outside the reparsed
// window (given in old-source coordinates), split into those preceding and
// those following it in document order. Definitions are rebuilt as owned
// values so that they are independent of old source positions.
func (d *editableDocument) outsideLinkDefinitions(
	wStartOld, wEndOld int,
) (before, after []LinkDefinition) {
	_ = ast.Walk(d.root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		ld, ok := n.(*ast.LinkReferenceDefinition)
		if !ok {
			return ast.WalkContinue, nil
		}
		pos := ld.Pos()
		if pos >= wStartOld && pos < wEndOld {
			return ast.WalkContinue, nil
		}
		def := NewLinkDefinition(
			ld.Label.Bytes(d.source),
			ld.Destination.Bytes(d.source),
			ld.Title.Bytes(d.source),
		)
		if pos < wStartOld {
			before = append(before, def)
		} else {
			after = append(after, def)
		}
		return ast.WalkContinue, nil
	})
	return before, after
}

// refreshDependentBlocks re-parses the inline content of reused blocks whose
// reference links may resolve differently after the edit.
func (d *editableDocument) refreshDependentBlocks(
	source []byte, fpc *parseContext, windowNodes map[ast.Node]bool,
) {
	changedLabels, addedLabels := changedLinkDefinitions(d.pc, fpc)
	if len(changedLabels) == 0 {
		return
	}

	owners := make(map[ast.Node]bool)
	_ = ast.Walk(d.root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		var ref *ast.ReferenceLink
		switch v := n.(type) {
		case *ast.Link:
			ref = v.Reference
		case *ast.Image:
			ref = v.Reference
		}
		if ref == nil {
			return ast.WalkContinue, nil
		}
		key := util.ToLinkReference(ref.Value.Bytes(source))
		if !changedLabels[key] {
			return ast.WalkContinue, nil
		}
		if owner := inlineOwnerBlock(n); owner != nil {
			owners[owner] = true
		}
		return ast.WalkContinue, nil
	})

	// A previously unresolved reference like "[foo]" has no Link node yet;
	// scan reused blocks for bracket expressions that could resolve to a
	// changed label so that adding a definition activates them.
	_ = ast.Walk(d.root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		bn, ok := n.(ast.BlockNode)
		if !ok || len(bn.Source()) == 0 || windowNodes[n] {
			return ast.WalkContinue, nil
		}
		// Only a brand new definition can activate previously literal
		// bracket text; changing an existing definition is handled above
		// through existing reference nodes, so extension-owned bracket
		// constructs (e.g. task list checkboxes) stay untouched.
		if len(addedLabels) == 0 {
			return ast.WalkContinue, nil
		}
		raw := text.NewLines(bn.Source()).Bytes(source)
		for _, key := range extractReferenceLabels(raw) {
			if addedLabels[key] {
				owners[n] = true
				break
			}
		}
		return ast.WalkContinue, nil
	})

	if len(owners) == 0 {
		return
	}
	reader := text.NewBlockReader(source, nil, d.mp.decoder)
	for owner := range owners {
		if windowNodes[owner] {
			continue
		}
		bn := owner.(ast.BlockNode)
		if len(bn.Source()) == 0 {
			continue
		}
		bn.RemoveChildren()
		d.mp.parseBlock(reader, bn, fpc)
	}
}

// changedLinkDefinitions compares the link definitions of two parse
// contexts and returns (1) the normalized label keys that differ (added,
// removed or changed destination/title) and (2) the keys that are newly
// present in the new context.
func changedLinkDefinitions(oldPC, newPC *parseContext) (changed, added map[string]bool) {
	changed = map[string]bool{}
	added = map[string]bool{}
	for key, newDef := range newPC.linkDefs {
		oldDef, oldOK := oldPC.linkDefs[key]
		switch {
		case !oldOK:
			changed[key] = true
			added[key] = true
		case !bytes.Equal(oldDef.Destination(), newDef.Destination()) ||
			!bytes.Equal(oldDef.Title(), newDef.Title()):
			changed[key] = true
		}
	}
	for key := range oldPC.linkDefs {
		if _, ok := newPC.linkDefs[key]; !ok {
			changed[key] = true
		}
	}
	return changed, added
}

// extractReferenceLabels scans raw block text for bracket expressions that
// could resolve as shortcut, collapsed or full reference links and returns
// the normalized labels. Inline links and unclosed brackets are ignored.
func extractReferenceLabels(raw []byte) []string {
	var labels []string
	for i := 0; i < len(raw); i++ {
		switch raw[i] {
		case '\\':
			i++
			continue
		case '[':
		default:
			continue
		}
		content, n := readBracket(raw, i)
		if n < 0 {
			continue
		}
		j := i + n
		switch {
		case j < len(raw) && raw[j] == '[':
			content2, n2 := readBracket(raw, j)
			if n2 < 0 {
				i = j
				continue
			}
			// Collapsed reference: [label][]. Otherwise full reference.
			label := content
			if len(bytes.TrimSpace(content2)) != 0 {
				label = content2
			}
			labels = append(labels, util.ToLinkReference(label))
			i = j + n2 - 1
		case j < len(raw) && raw[j] == '(':
			// Inline link: not a reference expression.
			i = j
		default:
			labels = append(labels, util.ToLinkReference(content))
			i = j - 1
		}
	}
	return labels
}

// readBracket returns the content and total length (including both
// delimiters) of the bracketed expression starting at raw[start] == '['.
// A single level of nested balanced brackets is allowed, as in CommonMark
// link labels. It returns -1 when no closing bracket exists.
func readBracket(raw []byte, start int) ([]byte, int) {
	depth := 0
	for i := start; i < len(raw); i++ {
		switch raw[i] {
		case '\\':
			i++
		case '[':
			depth++
			if depth > 2 {
				return nil, -1
			}
		case ']':
			depth--
			if depth == 0 {
				return raw[start+1 : i], i - start + 1
			}
		}
	}
	return nil, -1
}

// inlineOwnerBlock returns the nearest block ancestor that holds inline
// source segments (e.g. a Paragraph or Heading).
func inlineOwnerBlock(n ast.Node) ast.Node {
	for p := n.Parent(); p != nil; p = p.Parent() {
		if bn, ok := p.(ast.BlockNode); ok && len(bn.Source()) > 0 {
			return p
		}
	}
	return nil
}

// priorHeadingIDs returns the ids of headings located before the given
// source position, in document order.
func priorHeadingIDs(root ast.Node, beforePos int) []string {
	var out []string
	_ = ast.Walk(root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		h, ok := n.(*ast.Heading)
		if !ok || h.Pos() < 0 || h.Pos() >= beforePos {
			return ast.WalkContinue, nil
		}
		if id, ok := h.Attribute("id"); ok && !id.IsEmpty() {
			out = append(out, string(id.Bytes(nil)))
		}
		return ast.WalkContinue, nil
	})
	return out
}

// normalizeHeadingIDs walks headings in document order and guarantees that
// every id is unique the way a single full parse assigns them: a name already
// taken by an earlier heading gets a numeric suffix. The ids table is
// expected to already contain the ids preceding the reparsed window.
func normalizeHeadingIDs(root ast.Node, ids *IDs) {
	_ = ast.Walk(root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		h, ok := n.(*ast.Heading)
		if !ok {
			return ast.WalkContinue, nil
		}
		attr, exists := h.Attribute("id")
		if !exists || attr.IsEmpty() {
			return ast.WalkContinue, nil
		}
		name := string(attr.Bytes(nil))
		if ids.values != nil {
			if _, taken := ids.values[name]; taken {
				name = string(ids.Generate([]byte(name), ast.KindHeading))
				h.SetAttribute("id", text.NewMultiLineValueFromString(name, nil))
			}
		}
		ids.Put([]byte(name))
		return ast.WalkContinue, nil
	})
}

func childBlocks(parent ast.Node) []ast.Node {
	var out []ast.Node
	for c := parent.FirstChild(); c != nil; c = c.NextSibling() {
		out = append(out, c)
	}
	return out
}

// setLeftSpineBlankFlags sets HasBlankPreviousLines on the first-container
// spine of n (List -> ListItem -> Paragraph and similar), mirroring how the
// block parser propagates the previous-line blank state. Nodes created by
// paragraph transformers (such as GFM tables) always use their own fresh
// flags and are left untouched.
func setLeftSpineBlankFlags(n ast.Node, blank bool) {
	for n != nil {
		if !isCoreBlock(n) {
			return
		}
		n.(ast.BlockNode).SetBlankPreviousLines(blank)
		var firstBlock ast.Node
		for c := n.FirstChild(); c != nil; c = c.NextSibling() {
			if _, ok := c.(ast.BlockNode); ok {
				firstBlock = c
				break
			}
		}
		n = firstBlock
	}
}

// isCoreBlock reports whether n is a block node type produced directly by
// the core block parser (as opposed to a node synthesized by a paragraph
// transformer, whose blank flag is assigned by the transformer).
func isCoreBlock(n ast.Node) bool {
	switch n.(type) {
	case *ast.Document,
		*ast.Paragraph,
		*ast.Heading,
		*ast.List,
		*ast.ListItem,
		*ast.Blockquote,
		*ast.ThematicBreak,
		*ast.CodeBlock,
		*ast.HTMLBlock,
		*ast.LinkReferenceDefinition:
		return true
	}
	return false
}

func collectNodes(n ast.Node, set map[ast.Node]bool) {
	set[n] = true
	for c := n.FirstChild(); c != nil; c = c.NextSibling() {
		collectNodes(c, set)
	}
}

// blockStartOffset returns the earliest source position at which a block and
// its subtree start.
func blockStartOffset(n ast.Node) int {
	start := n.Pos()
	if bn, ok := n.(ast.BlockNode); ok {
		for _, seg := range bn.Source() {
			if start < 0 || seg.Start < start {
				start = seg.Start
			}
		}
	}
	return start
}

// buildLineStarts returns the byte offsets of every line start in source.
// Line 0 always starts at offset 0.
func buildLineStarts(source []byte) []int {
	starts := make([]int, 1, bytes.Count(source, []byte{'\n'})+1)
	for i := 0; i < len(source); i++ {
		if source[i] == '\n' && i+1 < len(source) {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// lineIndex returns the index of the line that contains the given offset.
func lineIndex(starts []int, offset int) int {
	i := sort.SearchInts(starts, offset)
	if i < len(starts) && starts[i] == offset {
		return i
	}
	return i - 1
}

func lineBytes(source []byte, starts []int, i int) []byte {
	if i+1 < len(starts) {
		return source[starts[i]:starts[i+1]]
	}
	return source[starts[i]:]
}

type fenceStateInfo struct {
	char   byte
	length int
}

// parseFenceMarker reports whether line opens a fenced code block.
// The logic mirrors the fenced code block parser.
func parseFenceMarker(line []byte) (fenceStateInfo, bool) {
	width, pos := util.IndentWidth(line, 0)
	if width > 3 || pos >= len(line) {
		return fenceStateInfo{}, false
	}
	c := line[pos]
	if c != '`' && c != '~' {
		return fenceStateInfo{}, false
	}
	i := pos
	for i < len(line) && line[i] == c {
		i++
	}
	length := i - pos
	if length < 3 {
		return fenceStateInfo{}, false
	}
	if c == '`' {
		// A backtick info string cannot contain a backtick.
		end := len(line)
		if end > 0 && line[end-1] == '\n' {
			end--
		}
		if bytes.IndexByte(line[i:end], '`') >= 0 {
			return fenceStateInfo{}, false
		}
	}
	return fenceStateInfo{char: c, length: length}, true
}

func isFenceCloser(line []byte, fence fenceStateInfo) bool {
	width, pos := util.IndentWidth(line, 0)
	if width >= 4 || pos >= len(line) {
		return false
	}
	if line[pos] != fence.char {
		return false
	}
	i := pos
	for i < len(line) && line[i] == fence.char {
		i++
	}
	if i-pos < fence.length {
		return false
	}
	return util.IsBlank(line[i:])
}

// oldListAt returns the old top-level List block whose range contains the
// given offset in the new source (mapped back to old coordinates), or nil.
func (d *editableDocument) oldListAt(newOffset, replEnd, delta int, bounds []int, children []ast.Node) *ast.List {
	if len(children) == 0 || newOffset < replEnd {
		return nil
	}
	oldOffset := newOffset - delta
	i := sort.Search(len(bounds), func(k int) bool { return bounds[k] > oldOffset }) - 1
	if i < 0 {
		return nil
	}
	l, _ := children[i].(*ast.List)
	return l
}

// oldBlockIncludableBackward reports whether the block owning prevLine (an
// adjacent non-blank line) may be pulled into the reparsed window when atLine
// becomes part of it. Paragraphs and link definitions always couple; a
// blockquote couples when atLine is plain prose that can be a lazy
// continuation (i.e. it does not itself start an interrupting block); other
// blocks keep a hard boundary.
func (d *editableDocument) oldBlockIncludableBackward(source []byte, starts []int, prevLine, atLine int,
	bounds []int, children []ast.Node) bool {
	if len(children) == 0 {
		return true
	}
	j := sort.Search(len(bounds), func(k int) bool { return bounds[k] > prevLine }) - 1
	if j < 0 {
		return true
	}
	switch children[j].(type) {
	case *ast.Paragraph, *ast.LinkReferenceDefinition:
		return true
	case *ast.List, *ast.Blockquote:
		// Lines that were already part of this block stay with it. Beyond
		// it, plain prose is a lazy paragraph continuation (adjacent lines
		// only; blank runs are handled separately), while a line that starts
		// an interrupting block terminates the container.
		if atLine >= bounds[j] && (j+1 >= len(bounds) || atLine < bounds[j+1]) {
			return true
		}
		return !lineStartsInterruptingBlock(lineBytes(source, starts, lineIndex(starts, atLine)))
	default:
		return false
	}
}

// oldListContinuesAt reports whether the line at atLine can be absorbed by
// the old list block whose line starts at prevLine (possibly across blank
// lines).
func (d *editableDocument) oldListContinuesAt(source []byte, starts []int, prevLine, atLine int,
	bounds []int, children []ast.Node) bool {
	if len(children) == 0 {
		return false
	}
	j := sort.Search(len(bounds), func(k int) bool { return bounds[k] > prevLine }) - 1
	if j < 0 {
		return false
	}
	list, ok := children[j].(*ast.List)
	if !ok {
		return false
	}
	return lineContinuesList(lineBytes(source, starts, lineIndex(starts, atLine)), list)
}

// lineContinuesList reports whether a non-blank line can belong to the given
// list after a blank line (a compatible new item marker or content indented
// to the list content column).
func lineContinuesList(line []byte, list *ast.List) bool {
	_, typ := parseListItem(line)
	if typ != notList && (typ == orderedList) == list.IsOrdered() {
		return true
	}
	w, _ := util.IndentWidth(line, 0)
	return w >= listContentOffset(list)
}

// lineStartsInterruptingBlock heuristically reports whether a line starts a
// block that can interrupt a blockquote paragraph (ATX heading, fence,
// thematic break or list item).
func lineStartsInterruptingBlock(line []byte) bool {
	w, pos := util.IndentWidth(line, 0)
	if w > 3 || pos >= len(line) {
		return false
	}
	t := line[pos:]
	switch {
	case t[0] == '#':
		// ATX heading: one to six hashes followed by whitespace/EOL.
		n := 0
		for n < len(t) && t[n] == '#' {
			n++
		}
		if n > 6 {
			return false
		}
		if n == len(t) || util.IsSpace(t[n]) {
			return true
		}
		return false
	case (t[0] == '`' || t[0] == '~'):
		_, ok := parseFenceMarker(line)
		return ok
	case t[0] == '-' || t[0] == '*' || t[0] == '+':
		// Only a real list item or thematic break interrupts.
		_, typ := parseListItem(line)
		return typ != notList || isThematicBreak(line, 0)
	case t[0] >= '0' && t[0] <= '9':
		_, typ := parseListItem(line)
		return typ != notList
	}
	return false
}

// expandWindow grows the initial reparsed range until its boundaries are safe
// block resynchronization points: a blank line (or document edge), except
// that lines belonging to an open fenced code block are always included up to
// the closing fence and blank lines inside loose lists are consumed together
// with the list content that follows them.
func expandWindow(source []byte, starts []int, startOff, endOff int,
	listAt func(offset int) *ast.List,
	canCrossBack func(prevLine, atLine int) bool,
	listAcrossBlank func(prevLine, atLine int) bool) (int, int) {
	k := lineIndex(starts, startOff)
	for k > 0 {
		if util.IsBlank(lineBytes(source, starts, k-1)) {
			// Blank lines are normally a hard stop, but a loose list
			// keeps owning content that follows a blank run: if a line
			// belonging to a list lies directly before the blanks and the
			// current first line continues that list, absorb the blanks.
			b := k - 1
			for b > 0 && util.IsBlank(lineBytes(source, starts, b-1)) {
				b--
			}
			if b >= 0 && listAcrossBlank != nil && listAcrossBlank(starts[b], starts[k]) {
				k = b
				continue
			}
			break
		}
		if canCrossBack != nil && !canCrossBack(starts[k-1], starts[k]) {
			break
		}
		k--
	}
	windowStart := starts[k]

	// Determine whether the initial window ends while a fence is open.
	var fence fenceStateInfo
	inFence := false
	for i := k; i < len(starts) && starts[i] < endOff; i++ {
		line := lineBytes(source, starts, i)
		if inFence {
			if isFenceCloser(line, fence) {
				inFence = false
			}
			continue
		}
		if info, ok := parseFenceMarker(line); ok {
			inFence = true
			fence = info
		}
	}

	s := sort.SearchInts(starts, endOff)
	for {
		if s >= len(starts) {
			return windowStart, len(source)
		}
		line := lineBytes(source, starts, s)
		if inFence {
			s++
			if isFenceCloser(line, fence) {
				inFence = false
			}
			continue
		}
		if util.IsBlank(line) {
			// Blank lines inside a loose list do not end the list; consume
			// the blank run together with every line that still belongs to
			// the list, then re-evaluate the boundary from there.
			if s > k && listAt != nil {
				if list := listAt(starts[s-1]); list != nil {
					s = consumeListTail(source, starts, s, list)
					continue
				}
			}
			// A blank line separates all other top-level blocks.
			return windowStart, starts[s]
		}
		// The candidate line is non-blank. It can only be structurally
		// coupled with the window when the window's own last line is also
		// non-blank (adjacent lines belong together: setext underline,
		// paragraph continuation, fence-less code and so on). An empty
		// window or one ending with a blank line is already resynchronized.
		if s <= k || util.IsBlank(lineBytes(source, starts, s-1)) {
			return windowStart, starts[s]
		}
		if info, ok := parseFenceMarker(line); ok {
			inFence = true
			fence = info
		}
		s++
	}
}

// consumeListTail advances over lines that still belong to a list once a
// blank line has been seen: blank lines, compatible new item markers and
// lines indented to the list content column. It returns the index of the
// first line that no longer belongs to the list.
func consumeListTail(source []byte, starts []int, s int, list *ast.List) int {
	offset := listContentOffset(list)
	ordered := list.IsOrdered()
	for s < len(starts) {
		line := lineBytes(source, starts, s)
		if util.IsBlank(line) {
			s++
			continue
		}
		_, typ := parseListItem(line)
		if typ != notList && (typ == orderedList) == ordered {
			s++
			continue
		}
		if w, _ := util.IndentWidth(line, 0); w >= offset {
			s++
			continue
		}
		break
	}
	return s
}

// listContentOffset returns the smallest content indentation of any item in
// the list: lines indented at least that much keep belonging to the list
// after a blank line.
func listContentOffset(list *ast.List) int {
	offset := 0
	_ = ast.Walk(list, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if li, ok := n.(*ast.ListItem); ok {
			o := li.Offset()
			if offset == 0 || o < offset {
				offset = o
			}
		}
		return ast.WalkContinue, nil
	})
	if offset == 0 {
		return 1
	}
	return offset
}

// windowFenceEnd inspects a reparsed window for fenced code blocks that run
// off the snippet end and returns the new-source offset up to which the
// window must be extended to include their closing fence (or the document
// end when no closer exists). open is false when every fence in the window
// is closed within it.
func windowFenceEnd(newSource []byte, newLines []int, wStart, wEnd int, snippet []byte, tmp ast.Node) (int, bool) {
	snippetStarts := buildLineStarts(snippet)
	furthest := wEnd
	found := false
	_ = ast.Walk(tmp, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		cb, ok := n.(*ast.CodeBlock)
		if !ok || cb.CodeBlockKind != ast.CodeBlockKindFenced {
			return ast.WalkContinue, nil
		}
		unclosed := false
		segs := cb.Value.Segments()
		if len(segs) > 0 {
			unclosed = segs[len(segs)-1].Stop >= len(snippet)
		} else if lineIndex(snippetStarts, cb.Pos()) >= len(snippetStarts)-1 {
			unclosed = true
		}
		if !unclosed {
			return ast.WalkContinue, nil
		}
		openerLine := lineIndex(newLines, wStart+cb.Pos())
		char, length, ok := scanFenceDelimiter(lineBytes(newSource, newLines, openerLine))
		if !ok {
			return ast.WalkContinue, nil
		}
		from := sort.SearchInts(newLines, wEnd)
		closer := findFenceCloseLine(newSource, newLines, from, char, length)
		end := len(newSource)
		if closer < len(newLines) {
			end = newLines[closer]
		}
		found = true
		if end > furthest {
			furthest = end
		}
		return ast.WalkContinue, nil
	})
	return furthest, found
}

// findFenceCloseLine returns the index of the line after the first line that
// closes a fence of the given kind at or after from, or len(starts) when no
// closer exists.
func findFenceCloseLine(source []byte, starts []int, from int, char byte, length int) int {
	for i := from; i < len(starts); i++ {
		c, n, ok := scanFenceDelimiter(lineBytes(source, starts, i))
		if ok && c == char && n >= length {
			return i + 1
		}
	}
	return len(starts)
}

// scanFenceDelimiter extracts a fence marker from a raw line, tolerating
// blockquote and list prefixes so that fences nested in containers are
// recognized too. It returns the fence character and run length.
func scanFenceDelimiter(line []byte) (byte, int, bool) {
	width, pos := util.IndentWidth(line, 0)
	if width > 12 || pos >= len(line) {
		return 0, 0, false
	}
	t := line[pos:]
	for {
		for len(t) > 0 && t[0] == '>' {
			t = t[1:]
			if len(t) > 0 && (t[0] == ' ' || t[0] == '\t') {
				t = t[1:]
			}
		}
		if len(t) > 0 && (t[0] == ' ' || t[0] == '\t') {
			t = t[1:]
			continue
		}
		break
	}
	if len(t) == 0 || (t[0] != '`' && t[0] != '~') {
		return 0, 0, false
	}
	c := t[0]
	i := 0
	for i < len(t) && t[i] == c {
		i++
	}
	if i < 3 {
		return 0, 0, false
	}
	return c, i, true
}

// shiftTree moves every source position stored in a tree by offset. Owned
// values are left untouched; the tree is modified in place.
func shiftTree(n ast.Node, offset int) {
	if offset != 0 {
		if shifter, ok := n.(interface{ ShiftSource(int) }); ok {
			shifter.ShiftSource(offset)
		}
		if _, isDoc := n.(*ast.Document); !isDoc {
			if pos := n.Pos(); pos >= 0 {
				n.SetPos(pos + offset)
			}
		}
		shiftNodeValues(n, offset)
		if attrs := n.Attributes(); len(attrs) > 0 {
			shifted := make([]ast.Attribute, len(attrs))
			for i, attr := range attrs {
				attr.Value = attr.Value.Shift(offset)
				shifted[i] = attr
			}
			n.RemoveAttributes()
			for _, attr := range shifted {
				n.SetAttribute(attr.Name, attr.Value)
			}
		}
	}
	for c := n.FirstChild(); c != nil; c = c.NextSibling() {
		shiftTree(c, offset)
	}
}

func shiftNodeValues(n ast.Node, offset int) {
	switch v := n.(type) {
	case *ast.Text:
		v.Value = v.Value.Shift(offset)
	case *ast.CodeSpan:
		switch cv := v.Value.(type) {
		case singleLineCodeSpanValue:
			cv.start += offset
			cv.stop += offset
			v.Value = cv
		case multiLineCodeSpanValue:
			indices := make([]text.Index, len(cv.indices))
			for i, idx := range cv.indices {
				idx.Start += offset
				idx.Stop += offset
				indices[i] = idx
			}
			v.Value = multiLineCodeSpanValue{indices: indices}
		}
	case *ast.Link:
		v.Destination = v.Destination.Shift(offset)
		v.Title = v.Title.Shift(offset)
		if v.Reference != nil {
			v.Reference.Value = v.Reference.Value.Shift(offset)
		}
	case *ast.Image:
		v.Destination = v.Destination.Shift(offset)
		v.Title = v.Title.Shift(offset)
		if v.Reference != nil {
			v.Reference.Value = v.Reference.Value.Shift(offset)
		}
	case *ast.AutoLink:
		v.Destination = v.Destination.Shift(offset)
		v.Label = v.Label.Shift(offset)
		v.Text = v.Text.Shift(offset)
	case *ast.RawHTML:
		v.Value = v.Value.Shift(offset)
	case *ast.CodeBlock:
		v.Info = v.Info.Shift(offset)
		v.Value.Shift(offset)
	case *ast.HTMLBlock:
		v.Value.Shift(offset)
	case *ast.LinkReferenceDefinition:
		v.Label = v.Label.Shift(offset)
		v.Destination = v.Destination.Shift(offset)
		v.Title = v.Title.Shift(offset)
	}
}
