package parser_test

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/yuin/goldmark/v2/ast"
	"github.com/yuin/goldmark/v2/extension"
	"github.com/yuin/goldmark/v2/parser"
	"github.com/yuin/goldmark/v2/renderer/html"
	"github.com/yuin/goldmark/v2/text"
	"github.com/yuin/goldmark/v2/util"
)

func renderHTML(t *testing.T, source []byte, doc ast.Node) string {
	t.Helper()
	var buf bytes.Buffer
	if err := html.New().Render(&buf, source, doc); err != nil {
		t.Fatalf("render error: %v", err)
	}
	return buf.String()
}

func dumpTree(t *testing.T, source []byte, doc ast.Node) string {
	t.Helper()
	var buf bytes.Buffer
	if err := doc.Dump(source).PrettyPrint(&buf, source); err != nil {
		t.Fatalf("dump error: %v", err)
	}
	return buf.String()
}

// checkEquivalent compares an editable document against a fresh full parse
// of the same source, both as rendered HTML and as dumped AST trees.
func checkEquivalent(p parser.Parser, source []byte, got ast.Node) error {
	full := p.Parse(source)

	var gotHTML, wantHTML bytes.Buffer
	if err := html.New().Render(&gotHTML, source, got); err != nil {
		return err
	}
	if err := html.New().Render(&wantHTML, source, full); err != nil {
		return err
	}
	if !bytes.Equal(gotHTML.Bytes(), wantHTML.Bytes()) {
		return fmt.Errorf("render mismatch\n--- got ---\n%s\n--- want ---\n%s", gotHTML.String(), wantHTML.String())
	}

	var gotDump, wantDump bytes.Buffer
	if err := got.Dump(source).PrettyPrint(&gotDump, source); err != nil {
		return err
	}
	if err := full.Dump(source).PrettyPrint(&wantDump, source); err != nil {
		return err
	}
	// Node properties are dumped from a map, so their order is random;
	// compare the sorted line multisets (HTML comparison above already pins
	// down node ordering and structure).
	gotLines := bytes.Split(gotDump.Bytes(), []byte("\n"))
	wantLines := bytes.Split(wantDump.Bytes(), []byte("\n"))
	slices.SortFunc(gotLines, bytes.Compare)
	slices.SortFunc(wantLines, bytes.Compare)
	if !bytes.Equal(bytes.Join(gotLines, []byte("\n")), bytes.Join(wantLines, []byte("\n"))) {
		return fmt.Errorf("ast dump mismatch\n--- got ---\n%s\n--- want ---\n%s", gotDump.String(), wantDump.String())
	}
	return nil
}

// assertEquivalent compares the editable session against a fresh full parse
// of the same source, both as rendered HTML and as dumped AST trees.
func assertEquivalent(t *testing.T, p parser.Parser, source []byte, doc parser.EditableDocument) {
	t.Helper()
	if err := checkEquivalent(p, source, doc.AST()); err != nil {
		t.Fatalf("%v\n--- source ---\n%s", err, source)
	}
}

func topLevelNodes(doc ast.Node) []ast.Node {
	var nodes []ast.Node
	for c := doc.FirstChild(); c != nil; c = c.NextSibling() {
		nodes = append(nodes, c)
	}
	return nodes
}

func TestEditableDocument_BasicRead(t *testing.T) {
	p := parser.New()
	src := []byte("# Title\n\nhello\n")
	d := parser.NewEditableDocument(p, src)

	if !bytes.Equal(d.Source(), src) {
		t.Fatalf("Source() = %q, want %q", d.Source(), src)
	}
	if d.AST() == nil || d.AST().Kind() != ast.KindDocument {
		t.Fatalf("AST() is not a document: %v", d.AST())
	}
	if got := renderHTML(t, src, d.AST()); got != "<h1>Title</h1>\n<p>hello</p>\n" {
		t.Fatalf("unexpected render: %q", got)
	}
}

func TestEditableDocument_SimpleEditIdentityAndEquivalence(t *testing.T) {
	p := parser.New()
	src := []byte("zero block\n\nfirst block\n\nsecond block\n\nthird block\n\nfourth block\n")
	d := parser.NewEditableDocument(p, src)

	before := topLevelNodes(d.AST())
	if len(before) != 5 {
		t.Fatalf("want 5 blocks, got %d", len(before))
	}

	// Edit the text of the second paragraph only.
	idx := bytes.Index(src, []byte("first block"))
	if err := d.ApplyEdit(idx, idx+len("first block"), []byte("FIRST BLOCK!")); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}

	after := topLevelNodes(d.AST())
	if len(after) != len(before) {
		t.Fatalf("block count changed: %d -> %d", len(before), len(after))
	}
	for i, n := range before {
		if i == 1 {
			if after[i] == n {
				t.Errorf("block %d should have been reparsed", i)
			}
			continue
		}
		if after[i] != n {
			t.Errorf("block %d identity lost after unrelated edit", i)
		}
	}

	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_PositionShift(t *testing.T) {
	p := parser.New()
	src := []byte("alpha\n\nbeta\n\ngamma\n")
	d := parser.NewEditableDocument(p, src)
	later := topLevelNodes(d.AST())[2]
	oldPos := later.Pos()

	// Insert text before every existing block.
	prefix := []byte("xxxxxxxxxxxx\n\n")
	if err := d.ApplyEdit(0, 0, prefix); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}

	full := p.Parse(d.Source())
	fullBlocks := topLevelNodes(full)
	movedLater := findNode(d.AST(), later)
	if movedLater == nil {
		t.Fatalf("session lost the later block identity")
	}
	if movedLater.Pos() != oldPos+len(prefix) {
		t.Errorf("later block Pos = %d, want %d", movedLater.Pos(), oldPos+len(prefix))
	}
	fullLater := fullBlocks[len(fullBlocks)-1]
	if movedLater.Pos() != fullLater.Pos() {
		t.Errorf("later block Pos = %d, full parse gives %d", movedLater.Pos(), fullLater.Pos())
	}
	assertEquivalent(t, p, d.Source(), d)

	// Delete the prefix again; positions must return to the originals.
	if err := d.ApplyEdit(0, len(prefix), nil); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}
	if moved := findNode(d.AST(), later); moved == nil {
		t.Fatalf("later block identity lost after prefix delete")
	} else if moved.Pos() != oldPos {
		t.Errorf("later block Pos after delete = %d, want %d", moved.Pos(), oldPos)
	}
	assertEquivalent(t, p, d.Source(), d)
}

// findNode looks for the exact node pointer anywhere in the tree.
func findNode(root ast.Node, target ast.Node) ast.Node {
	var found ast.Node
	_ = ast.Walk(root, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if entering && n == target {
			found = n
			return ast.WalkStop, nil
		}
		return ast.WalkContinue, nil
	})
	return found
}

func TestEditableDocument_EmptyAndNoopEdits(t *testing.T) {
	p := parser.New()
	src := []byte("one\n\ntwo\n")
	d := parser.NewEditableDocument(p, src)
	before := d.AST()

	if err := d.ApplyEdit(0, 0, nil); err != nil {
		t.Fatalf("empty edit: %v", err)
	}
	if d.AST() != before {
		t.Errorf("empty edit replaced the document tree")
	}
	if err := d.ApplyEdit(len(src), len(src), []byte{}); err != nil {
		t.Fatalf("empty edit at EOF: %v", err)
	}
	if d.AST() != before {
		t.Errorf("empty edit at EOF replaced the document tree")
	}
	// Replacing a range with identical bytes is also a no-op.
	idx := bytes.Index(src, []byte("two"))
	if err := d.ApplyEdit(idx, idx+3, []byte("two")); err != nil {
		t.Fatalf("identical replacement: %v", err)
	}
	if d.AST() != before {
		t.Errorf("identical replacement replaced the document tree")
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_OutOfRangeKeepsState(t *testing.T) {
	p := parser.New()
	src := []byte("hello world\n")
	d := parser.NewEditableDocument(p, src)
	before := d.AST()
	beforeSource := d.Source()

	cases := []struct {
		start, end int
	}{
		{-1, 0},
		{0, -1},
		{0, len(src) + 1},
		{len(src) + 1, len(src) + 2},
		{5, 2},
	}
	for _, c := range cases {
		err := d.ApplyEdit(c.start, c.end, []byte("x"))
		if err == nil {
			t.Errorf("ApplyEdit(%d,%d) expected error, got nil", c.start, c.end)
		}
		if !errors.Is(err, parser.ErrInvalidEditRange) {
			t.Errorf("ApplyEdit(%d,%d) error = %v, want ErrInvalidEditRange", c.start, c.end, err)
		}
		if d.AST() != before {
			t.Errorf("ApplyEdit(%d,%d) replaced the document tree", c.start, c.end)
		}
		if !bytes.Equal(d.Source(), beforeSource) {
			t.Errorf("ApplyEdit(%d,%d) changed the source", c.start, c.end)
		}
	}

	// The session is still usable after rejected edits.
	if err := d.ApplyEdit(0, 5, []byte("HELLO")); err != nil {
		t.Fatalf("valid edit after rejected edits: %v", err)
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_DefensiveCopies(t *testing.T) {
	p := parser.New()
	src := []byte("hello\n")
	d := parser.NewEditableDocument(p, src)
	src[0] = 'X'
	if got := d.Source(); !bytes.Equal(got, []byte("hello\n")) {
		t.Errorf("session affected by mutation of constructor input: %q", got)
	}

	repl := []byte("HELLO")
	if err := d.ApplyEdit(0, 5, repl); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}
	repl[0] = 'z'
	if got := d.Source(); !bytes.Equal(got, []byte("HELLO\n")) {
		t.Errorf("session affected by mutation of replacement: %q", got)
	}

	out := d.Source()
	out[0] = '!'
	if got := d.Source(); got[0] != 'H' {
		t.Errorf("session affected by mutation of Source() result: %q", got)
	}
}

// TestEditableDocument_WithOptionsAndExtensions exercises sessions created
// with non-default parser options.
func TestEditableDocument_WithOptionsAndExtensions(t *testing.T) {
	p := parser.New(parser.WithAutoHeadingID())
	src := []byte("# Heading\n\npara one\n\npara two\n")
	d := parser.NewEditableDocument(p, src)
	if err := d.ApplyEdit(9, 9, []byte(" Two")); err != nil {
		t.Fatal(err)
	}
	assertEquivalent(t, p, d.Source(), d)

	// A brand new heading introduced by an edit must get a stable id even
	// though the id table is rebuilt per edit.
	if err := d.ApplyEdit(0, 0, []byte("## Intro\n\n")); err != nil {
		t.Fatal(err)
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_WithGFMExtension(t *testing.T) {
	p := parser.New(parser.WithExtensions(extension.GFMParser))
	src := []byte("# H\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\n" +
		"- [x] done\n- [ ] todo\n\n" +
		"~~strike~~ and [x][x]\n\n[x]: /u\n\n```\ncode\n```\n")
	d := parser.NewEditableDocument(p, src)
	replace := func(marker, with string) {
		t.Helper()
		cur := d.Source()
		idx := bytes.Index(cur, []byte(marker))
		if idx < 0 {
			t.Fatalf("marker %q not found in %q", marker, cur)
		}
		if err := d.ApplyEdit(idx, idx+len(marker), []byte(with)); err != nil {
			t.Fatal(err)
		}
		assertEquivalent(t, p, d.Source(), d)
	}
	insert := func(marker string, with []byte) {
		t.Helper()
		idx := bytes.Index(d.Source(), []byte(marker))
		if idx < 0 {
			t.Fatalf("marker %q not found", marker)
		}
		if err := d.ApplyEdit(idx, idx, with); err != nil {
			t.Fatal(err)
		}
		assertEquivalent(t, p, d.Source(), d)
	}
	replace("| a | b |", "| A | B |")
	replace("- [ ] todo", "- [x] todo")
	replace("/u", "/v")
	insert("| - | - |", []byte("| - | - |\n| 9 | 9 |\n"))
	insert("# H", []byte("## Intro\n"))
	insert("~~strike~~", []byte("\n\n"))
	insert("```\ncode\n```\n", []byte("\nnew tail\n"))
	replace("| 1 | 2 |", "x")
}

type countingBlockParser struct {
	inner  parser.BlockParser
	opens  int
	closes int
}

func (c *countingBlockParser) Trigger() []byte { return c.inner.Trigger() }

func (c *countingBlockParser) Open(parent ast.Node, reader text.Reader, pc parser.Context) (ast.Node, parser.State) {
	node, state := c.inner.Open(parent, reader, pc)
	if node != nil {
		c.opens++
	}
	return node, state
}

func (c *countingBlockParser) Continue(node ast.Node, reader text.Reader, pc parser.Context) parser.State {
	return c.inner.Continue(node, reader, pc)
}

func (c *countingBlockParser) Close(node ast.Node, reader text.Reader, pc parser.Context) {
	c.closes++
	c.inner.Close(node, reader, pc)
}

func (c *countingBlockParser) CanInterruptParagraph() bool { return c.inner.CanInterruptParagraph() }
func (c *countingBlockParser) CanAcceptIndentedLine() bool {
	return c.inner.CanAcceptIndentedLine()
}

func newCountingParser() (*countingBlockParser, parser.Parser) {
	cp := &countingBlockParser{inner: parser.NewParagraphParser()}
	p := parser.New(
		parser.WithDefaultParsers(false),
		parser.WithBlockParsers(
			util.Prioritized(parser.NewSetextHeadingParser(), 100),
			util.Prioritized(parser.NewThematicBreakParser(), 200),
			util.Prioritized(parser.NewListParser(), 300),
			util.Prioritized(parser.NewListItemParser(), 400),
			util.Prioritized(parser.NewCodeBlockParser(), 500),
			util.Prioritized(parser.NewATXHeadingParser(), 600),
			util.Prioritized(parser.NewFencedCodeBlockParser(), 700),
			util.Prioritized(parser.NewBlockquoteParser(), 800),
			util.Prioritized(parser.NewHTMLBlockParser(), 900),
			util.Prioritized[parser.BlockParser](cp, 1000),
		),
	)
	return cp, p
}

func TestEditableDocument_BlockParserCallbacksSkipped(t *testing.T) {
	cp, p := newCountingParser()
	src := []byte("block zero\n\nblock one\n\nblock two\n\nblock three\n\nblock four\n")
	d := parser.NewEditableDocument(p, src)
	before := topLevelNodes(d.AST())

	opensAtStart := cp.opens
	idx := bytes.Index(src, []byte("block two"))
	if err := d.ApplyEdit(idx, idx+len("block two"), []byte("BLOCK TWO")); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}

	// Only the edited paragraph is reopened; all other blocks keep identity.
	if got, want := cp.opens-opensAtStart, 1; got != want {
		t.Errorf("paragraph Open callbacks: got %d, want %d", got, want)
	}
	after := topLevelNodes(d.AST())
	for i, n := range before {
		if i == 2 {
			if after[i] == n {
				t.Errorf("edited block %d unexpectedly reused", i)
			}
			continue
		}
		if after[i] != n {
			t.Errorf("block %d was reparsed despite being unaffected", i)
		}
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_LinkDefinitions(t *testing.T) {
	p := parser.New()
	src := []byte("[a]: /old \"Old title\"\n[b]: /b\n\n" +
		"Use [a][a] and [b][b] in this paragraph.\n\n" +
		"This paragraph has no references at all.\n")
	d := parser.NewEditableDocument(p, src)

	root := d.AST()
	var dependent, unrelated ast.Node
	for _, c := range topLevelNodes(root) {
		if c.Kind() == ast.KindParagraph {
			if bytes.Contains([]byte(renderHTML(t, src, c)), []byte("/old")) {
				dependent = c
			} else {
				unrelated = c
			}
		}
	}
	if dependent == nil || unrelated == nil {
		t.Fatalf("could not locate test paragraphs")
	}

	// Change destination of definition [a].
	dst := bytes.Index(src, []byte("/old"))
	if err := d.ApplyEdit(dst, dst+len("/old"), []byte("/new")); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}

	html := renderHTML(t, d.Source(), d.AST())
	if !strings.Contains(html, `href="/new"`) {
		t.Errorf("dependent link not updated:\n%s", html)
	}
	if strings.Contains(html, `href="/old"`) {
		t.Errorf("old destination still rendered:\n%s", html)
	}
	if !strings.Contains(html, `href="/b"`) {
		t.Errorf("unrelated link destination lost:\n%s", html)
	}
	if topLevelNodes(d.AST())[3] != unrelated {
		t.Errorf("unrelated paragraph was reparsed")
	}
	// The dependent paragraph block may keep its identity, but its link
	// destination must have been refreshed.
	depHTML := renderHTML(t, d.Source(), topLevelNodes(d.AST())[2])
	if !strings.Contains(depHTML, `href="/new"`) {
		t.Errorf("dependent block not refreshed: %s", depHTML)
	}
	assertEquivalent(t, p, d.Source(), d)

	// Remove the definition: reference links fall back to literal text, and
	// the previously independent paragraph still keeps its identity.
	if err := d.ApplyEdit(0, len("[a]: /new \"Old title\"\n"), nil); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}
	html = renderHTML(t, d.Source(), d.AST())
	if strings.Contains(html, `href="/new"`) {
		t.Errorf("removed definition still resolves:\n%s", html)
	}
	if !strings.Contains(html, "[a][a]") {
		t.Errorf("reference should render as literal text:\n%s", html)
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_AddLinkDefinitionActivatesReference(t *testing.T) {
	p := parser.New()
	src := []byte("shortcut [foo] in text.\n")
	d := parser.NewEditableDocument(p, src)
	if strings.Contains(renderHTML(t, src, d.AST()), "<a ") {
		t.Fatalf("precondition: no link expected")
	}

	// Insert a definition before the paragraph.
	def := []byte("[foo]: /url\n\n")
	if err := d.ApplyEdit(0, 0, def); err != nil {
		t.Fatalf("ApplyEdit: %v", err)
	}
	html := renderHTML(t, d.Source(), d.AST())
	if !strings.Contains(html, `href="/url"`) {
		t.Errorf("shortcut reference was not activated:\n%s", html)
	}
	assertEquivalent(t, p, d.Source(), d)
}

func TestEditableDocument_ContinuousEdits(t *testing.T) {
	p := parser.New()
	d := parser.NewEditableDocument(p, []byte("start\n"))

	type edit struct {
		start, end int
		text       string
	}
	edits := []edit{
		{5, 5, " block"},
		{0, 0, "# Heading\n\n"},
		{11, 11, "\n\nsecond"},
		{0, 9, "## BIG"},
		{0, 0, "> quoted\n>\n"},
		{20, 20, " tail"},
		{0, 0, "- "},
		{3, 3, "\n\n```\ncode\n```\n"},
	}
	for i, e := range edits {
		if e.end > len(d.Source()) {
			t.Fatalf("edit %d: end %d out of range (len %d)", i, e.end, len(d.Source()))
		}
		if err := d.ApplyEdit(e.start, e.end, []byte(e.text)); err != nil {
			t.Fatalf("edit %d: %v", i, err)
		}
		assertEquivalent(t, p, d.Source(), d)
	}
}

func TestEditableDocument_BoundaryStability(t *testing.T) {
	p := parser.New()

	cases := []struct {
		name   string
		source string
		edit   func(src []byte) (start, end int, text []byte)
	}{
		{
			name:   "list item text",
			source: "- alpha\n- beta\n- gamma\n\nparagraph after\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("beta"))
				return i, i + 4, []byte("BETA")
			},
		},
		{
			name:   "list becomes loose",
			source: "- a\n- b\n- c\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("- b"))
				return i, i, []byte("\n")
			},
		},
		{
			name:   "blockquote add line",
			source: "> quote one\n> quote two\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("quote two"))
				return i, i, []byte("extended ")
			},
		},
		{
			name:   "blockquote marker removed",
			source: "> quote one\n> quote two\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("> quote two"))
				return i, i + 2, []byte("")
			},
		},
		{
			name:   "fenced content edit",
			source: "before\n\n```\ncode one\ncode two\n```\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("code two"))
				return i, i + 8, []byte("CODE TWO")
			},
		},
		{
			name:   "fence opener removed swallows paragraph",
			source: "before\n\n```\ncode\n```\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("```\ncode"))
				return i, i + 3, []byte("")
			},
		},
		{
			name:   "new fence opens around text",
			source: "before\n\nplain text\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("plain text"))
				return i, i, []byte("```\n")
			},
		},
		{
			name:   "setext underline broken by blank line",
			source: "heading text\n=====\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("====="))
				return i, i, []byte("\n")
			},
		},
		{
			name:   "setext underline created",
			source: "heading text\nplain underline\n\nafter\n",
			edit: func(src []byte) (int, int, []byte) {
				i := bytes.Index(src, []byte("plain underline"))
				return i, i + len("plain underline"), []byte("=====")
			},
		},
		{
			name:   "edit at EOF without trailing newline",
			source: "para one\n\npara two",
			edit: func(src []byte) (int, int, []byte) {
				return len(src), len(src), []byte(" extended")
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := []byte(tc.source)
			d := parser.NewEditableDocument(p, src)
			start, end, repl := tc.edit(src)
			if err := d.ApplyEdit(start, end, repl); err != nil {
				t.Fatalf("ApplyEdit: %v", err)
			}
			assertEquivalent(t, p, d.Source(), d)
		})
	}
}

func TestEditableDocument_FuzzLikeEdits(t *testing.T) {
	p := parser.New()
	src := []byte("# Heading\n\n" +
		"A paragraph with *emphasis*, `code` and [a link][ref].\n\n" +
		"[ref]: https://example.com \"Title\"\n\n" +
		"- list one\n- list two\n  - nested item\n- list three\n\n" +
		"> blockquote line 1\n> line 2 with **strong**\n\n" +
		"```go\nfunc main() {}\n```\n\n" +
		"1. ordered one\n2. ordered two\n\n" +
		"final paragraph\n")

	for _, seed := range []int64{42, 7, 99, 1234567, 0xdeadbeef, 555, 987654} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			runFuzzEdits(t, p, src, seed)
		})
	}
}

func TestEditableDocument_FuzzRandomDocument(t *testing.T) {
	tokens := []string{
		"a", "b", " ", "  ", "\t", "*", "**", "#", "## ", ">", "> ", "- ",
		"1. ", "`", "``", "```", "~~~", "x", "\n", "\n\n", "\r\n",
		"[ref]", "[ref]: /u\n", "===", "---", "(", ")", "]",
		"http://x.test ", "text ",
	}
	for _, seed := range []int64{1, 2, 3} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			p := parser.New()
			rng := rand.New(rand.NewSource(seed))
			var initial []byte
			for range 80 {
				initial = append(initial, tokens[rng.Intn(len(tokens))]...)
			}
			d := parser.NewEditableDocument(p, initial)
			current := append([]byte(nil), initial...)
			for i := 0; i < 300; i++ {
				start := rng.Intn(len(current) + 1)
				end := start
				if rng.Intn(2) == 0 && start < len(current) {
					end = start + rng.Intn(8)
					if end > len(current) {
						end = len(current)
					}
				}
				repl := []byte(tokens[rng.Intn(len(tokens))])
				if err := d.ApplyEdit(start, end, repl); err != nil {
					t.Fatalf("iteration %d: %v", i, err)
				}
				next := make([]byte, 0, len(current)-end+start+len(repl))
				next = append(next, current[:start]...)
				next = append(next, repl...)
				next = append(next, current[end:]...)
				current = next
				if err := checkEquivalent(p, current, d.AST()); err != nil {
					t.Fatalf("iteration %d: %v\n--- source ---\n%s", i, err, current)
				}
			}
		})
	}
}

func runFuzzEdits(t *testing.T, p parser.Parser, src []byte, seed int64) {
	t.Helper()
	d := parser.NewEditableDocument(p, src)
	current := append([]byte(nil), src...)

	rng := rand.New(rand.NewSource(seed))
	replacements := []string{
		"", " ", "x", "\n", "\n\n", "**", "## ", "> ", "- ", "`", "``",
		"[ref]: https://other.example.com\n", "=====\n", "```\n", "text",
	}
	var history []string
	for i := 0; i < 400; i++ {
		start := rng.Intn(len(current) + 1)
		end := start
		if rng.Intn(2) == 0 && start < len(current) {
			end = start + 1 + rng.Intn(6)
			if end > len(current) {
				end = len(current)
			}
		}
		repl := []byte(replacements[rng.Intn(len(replacements))])
		history = append(history, fmt.Sprintf("ApplyEdit(%d, %d, %q)", start, end, repl))
		if err := d.ApplyEdit(start, end, repl); err != nil {
			t.Fatalf("iteration %d: %v\n%s", i, err, strings.Join(history, "\n"))
		}
		next := make([]byte, 0, len(current)-end+start+len(repl))
		next = append(next, current[:start]...)
		next = append(next, repl...)
		next = append(next, current[end:]...)
		current = next
		if !bytes.Equal(d.Source(), current) {
			t.Fatalf("iteration %d: source mismatch", i)
		}
		if err := checkEquivalent(p, current, d.AST()); err != nil {
			t.Fatalf("iteration %d (seed %d): %v\n%s\n--- source ---\n%s", i, seed, err, strings.Join(history, "\n"), current)
		}
	}
}

func TestEditableDocument_FuzzHardEdits(t *testing.T) {
	p := parser.New()
	src := []byte("# Title\r\n\r\n" +
		"Lorem *ipsum* dolor `sit` [x][x],\r\namet **consectetur**.\r\n\r\n" +
		"[x]: https://x.test \"X\"\r\n[y]: https://y.test\r\n\r\n" +
		"1. one\r\n2. two\r\n   1. nested\r\n3. three\r\n\r\n" +
		"> quoted\r\n> line\r\n\r\n" +
		"~~~\ntilde fence\n~~~\n\n" +
		"\tindented code\n\n" +
		"---\n\ntail\n")

	for _, seed := range []int64{11, 222, 3333, 44444} {
		t.Run(fmt.Sprintf("seed%d", seed), func(t *testing.T) {
			d := parser.NewEditableDocument(p, src)
			current := append([]byte(nil), src...)
			rng := rand.New(rand.NewSource(seed))
			replacements := []string{
				"", " ", "x", "\n", "\r\n", "\r", "\t", "**", "## ", "> ", "- ",
				"1. ", "`", "``", "~~~", "[x]: /z\n", "=====\n", "```", "\n\n",
			}
			for i := 0; i < 250; i++ {
				start := rng.Intn(len(current) + 1)
				end := start
				if rng.Intn(2) == 0 && start < len(current) {
					end = start + rng.Intn(10)
					if end > len(current) {
						end = len(current)
					}
				}
				repl := []byte(replacements[rng.Intn(len(replacements))])
				if err := d.ApplyEdit(start, end, repl); err != nil {
					t.Fatalf("iteration %d: %v", i, err)
				}
				next := make([]byte, 0, len(current)-end+start+len(repl))
				next = append(next, current[:start]...)
				next = append(next, repl...)
				next = append(next, current[end:]...)
				current = next
				if err := checkEquivalent(p, current, d.AST()); err != nil {
					t.Fatalf("iteration %d: %v\n--- source ---\n%s", i, err, current)
				}
			}
		})
	}
}

type countingASTTransformer struct {
	calls int
}

func (t *countingASTTransformer) Transform(node *ast.Document, _ text.Reader, _ parser.Context) {
	t.calls++
	// Every invocation must see the whole document: the first and last
	// blocks are always present.
	if node.FirstChild() == nil || node.LastChild() == nil {
		panic("transformer saw an incomplete document")
	}
}

func TestEditableDocument_ASTTransformerRunsOnFullTree(t *testing.T) {
	tr := &countingASTTransformer{}
	p := parser.New(parser.WithASTTransformers(util.Prioritized[parser.ASTTransformer](tr, 100)))
	src := []byte("first block\n\nmiddle block\n\nlast block\n")
	d := parser.NewEditableDocument(p, src)
	if tr.calls != 1 {
		t.Fatalf("transformer calls after construction: %d, want 1", tr.calls)
	}

	want := "middle block"
	for i := 1; i <= 3; i++ {
		cur := d.Source()
		idx := bytes.Index(cur, []byte(want))
		next := "middle edit" + string(rune('0'+i))
		if err := d.ApplyEdit(idx, idx+len(want), []byte(next)); err != nil {
			t.Fatalf("ApplyEdit: %v", err)
		}
		want = next
		if tr.calls != i+1 {
			t.Errorf("after edit %d transformer calls = %d, want %d", i, tr.calls, i+1)
		}
		if topLevelNodes(d.AST())[0].Kind() != ast.KindParagraph ||
			topLevelNodes(d.AST())[2].Kind() != ast.KindParagraph {
			t.Errorf("transformer did not see the full document on edit %d", i)
		}
	}
	assertEquivalent(t, p, d.Source(), d)
}
