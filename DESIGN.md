# Tensor Studio interface

## Purpose and scope

Tensor Studio helps researchers and engineers inspect a supported model, choose
compression methods, and compare the reloaded artifact with its original.
The interface should make the next action obvious while keeping research limits
visible. The existing warm teal identity is retained; this is an operating tool,
not a marketing page. No frontend framework or external font service is required.

The guided setup has three steps: **Model & data → Methods → Review & run**.
Choosing a step changes only the visible section. Back preserves all settings.
Only the final action starts a worker. Step numbers describe setup order, not
experiment progress. Unsupported capabilities remain disabled with an explanation.

## Layout, hierarchy and navigation

Desktop uses a 240 px sidebar for New run, previous runs and connection status.
The main column contains a short heading, a setup step navigation and one active
section with an 840 px maximum width. The section heading, fields, readiness hint
and final navigation form a consistent reading sequence. Related controls use
two columns when space permits; explanatory prose stays below 72 characters wide.

Example screens:

1. **Model & data:** choose compression or inspection, then a preset. The quick
   demo has its safe default model settings collapsed. Custom model presets open
   the settings needed for trained weights, datasets or pinned Hugging Face
   revisions. A synthetic-data notice explains the evidence boundary. Continue
   names its destination and starts no computation.
2. **Methods:** select SVD, packed int8, supported gated-MLP pruning, or a supported
   combination. Only selected method parameters appear. The language preset
   separates its fixed reference study from a custom recipe. Inspection instead
   explains the loss-sensitivity plot. Back and Review your run remain predictable.
3. **Review & run:** a definition list shows model, task, data, methods and their
   key parameters, compute, downloads, recovery and quality limit. Quality/size
   settings and full JSON are optional disclosures. This is the last editable
   checkpoint before the clearly named final action.
4. **Running:** show the actual worker stage and an activity indicator, without
   a fabricated percentage or time estimate. The user can return through history.
   Technical logs are available without dominating the screen.
5. **Results:** lead with the outcome and evidence limitations, then compare
   serialized size, quality and latency. Detailed tables, layer-relevance bars,
   validation trials and artifacts follow when available. Failed, slower and
   synthetic outcomes remain visible. Use this configuration returns to setup.

```text
Desktop: [New run / history] | Heading
                             Model & data — Methods — Review & run
                             Active section
                             Readiness hint
                             Back                 Next / Start

Mobile:  Brand                         New run
         Previous runs disclosure
         Heading
         Model & data | Methods | Review & run
         Active section, stacked fields
         Readiness hint
         Back                     Next / Start
```

The primary action is the lowest item in the active flow rather than a floating
toolbar: it cannot obscure fields at high zoom or cover a phone's keyboard.

## Interaction patterns

- **Hover:** buttons and selection rows change background/border in 160 ms.
  Hover communicates interactivity; no information depends on it.
- **Tap/click:** the entire labeled selection row is interactive. Native inputs,
  selectors, file pickers and disclosures keep familiar behavior. Selected rows
  retain both a checked control and a stronger border.
- **Keyboard:** Tab follows visible DOM order; Enter/Space activate native
  controls. Enter in an early step advances setup rather than starting a job.
  A step change focuses its heading; revisiting settings preserves values.
  History refreshes preserve focus on the same run.
- **Validation:** text/numeric settings validate as they change. Missing weights,
  unpinned revisions, empty methods and invalid numeric ranges block the next
  action with a specific readiness message. Invalid JSON stays reachable for
  repair rather than trapping the user behind a disabled step.
- **Transitions:** one 180 ms section reveal explains a step change. A subtle
  activity pulse communicates running state. There is no entrance choreography.
- **Drag:** no custom drag interaction is required. Model uploads use the native
  file picker; plots and configuration remain operable without precise dragging.
- **Downloads:** artifact links have accessible Download names and preserve the
  actual artifact filename. The final review explicitly states download permission
  for missing model/data assets.

## Visual guidelines

| Token | Value | Purpose |
| --- | --- | --- |
| Paper | `#f7f8f4` | Warm, quiet workspace background |
| Surface | `#ffffff` | Editable content and readable evidence |
| Sidebar | `#edf2eb` | Secondary navigation |
| Ink | `#203a32` | Primary text |
| Muted | `#506259` | Helper text with readable contrast |
| Action | `#155f56` | Primary actions, selection and focus |
| Selection | `#eaf3ed` | Selected controls |
| Control border | `#819188` | Input/control boundaries |
| Success | `#205637` on `#eef6ef` | Passed checks, accompanied by words |
| Warning | `#72520e` on `#fff5db` | Evidence limits and unmet requirements |
| Error | `#973b2c` on `#fff0ec` | Failures and recovery instructions |

Use an 8 px spacing rhythm: 8 px between labels and controls, 12–16 px inside
related groups, 24 px between groups, and 28–40 px for section/workspace padding.
The typography uses Segoe UI/system sans, available offline: 16 px inputs/body,
14–15 px labels/actions, 13 px helper text, 22 px section headings and 32 px page
headings (28 px on mobile). Code alone uses monospace; measurements use tabular
numerals. Plain language explains SVD energy and NLL without changing their meaning.

Panels use a 12 px radius and a quiet border. Buttons/inputs use 8 px radii;
selection rows use 10 px. Avoid decorative shadows, gradients, nested cards and
uppercase labels. Small authored SVG icons share one stroke vocabulary; icons
support labels rather than replacing them.

## Accessibility

Body/helper text targets WCAG AA contrast of at least 4.5:1; large text requires
3:1, as do meaningful control boundaries. Main controls are at least 48 px high;
text links/actions provide a 44 px target. Selected, disabled, warning and error
states include words or native control state rather than color alone.

Keep persistent labels, fieldsets/legends, a skip link, semantic headings and
`aria-current="step"` on setup navigation. Hidden steps leave the keyboard order.
Errors use `role="alert"`; readiness and genuine worker-stage changes use polite
status announcements. Repeated polling does not reannounce unchanged stage text.
Results tables retain captions and keyboard-focusable scrolling containers.
Respect `prefers-reduced-motion` by removing animations and transitions.

Browser/DOM checks cover keyboard focus, accessible names, validation and narrow
layouts. They do not certify all assistive technologies; test NVDA/VoiceOver and
actual touch devices before claiming a complete accessibility audit.

## Responsive behavior

- **Desktop, above 980 px:** fixed sidebar, centered content, two-column related
  controls and three parallel result summaries.
- **Tablet, 621–980 px:** sidebar becomes a top header and history disclosure;
  the same step order and navigation remain. The content uses available width.
- **Mobile, up to 620 px:** steps remain visible, fields/selection rows stack,
  helper text wraps and input text stays 16 px. Review labels and values share a
  compact two-column definition list. Result summaries stack; tables scroll
  horizontally within their own container rather than widening the page.

## Rationale and optional enhancements

Guided steps reduce simultaneous decisions. A review checkpoint makes computation
and downloads intentional. Progressive disclosure preserves expert controls
without making them prerequisites for the quick demo. Larger type and quieter
hierarchy improve scanability. Native controls, semantic markup and no external
assets keep the demo predictable, accessible and usable offline. Honest evidence
labels protect research credibility without burying the user in warnings.

Implemented small comforts include ready presets, preserved settings, immediate
validation, responsive history, a subtle step reveal and a readable final summary.
Potential later enhancements are saved named recipes, stronger inline error
association and an accessible native GPU-study result viewer. Add these only when
they improve an already verified compression workflow; do not imply new methods
or performance results through presentation.

## Verify the interface yourself

Run `docker compose up -d --build`, then open `http://localhost:7860`.

1. Start the quick demo. Advance through Model, Methods and Review using Enter;
   confirm no worker starts until Compress & compare.
2. In Methods, deselect every method; Review must become unavailable. Select
   SVD, enter energy above 1; the next action must remain blocked. Restore 0.5.
3. Review and go Back; the energy must remain unchanged. Complete the synthetic
   demo and inspect its size, quality, latency, evidence notice and downloads.
4. Choose your own vision classifier without weights. Continue must explain the
   missing trained model. Choose the language preset and enter `main` as the model
   revision; Continue must require a full commit. Do not start a real language run
   unless its downloads and compute are intentional.
5. Choose the custom language recipe, select all supported methods and inspect
   the final summary and JSON. It must contain the custom recipe rather than the
   fixed language study. Changing the workflow must remain explicit.
6. In Review, enter invalid JSON. Starting is blocked and the editor remains
   reachable for repair. Restore valid JSON before continuing.
7. Resize to 1440, 820 and 390 px. Check the sidebar/history transformation,
   stacked fields, unchanged values, contained table scrolling and visible actions.
8. Use Tab/Shift+Tab, Space and Enter, then enable reduced motion. Check focus
   visibility and repeat the flow without requiring a mouse or animation.

Development verification on 2026-10-14: the CPU suite passed 210 tests with 6
Git/Bash-dependent skips and 18 existing warnings. JavaScript syntax and whitespace
checks passed. Browser assertions exercised missing assets, pins, method/range
guards, retained recipes, JSON repair, keyboard step changes and phone/tablet
widths. Docker built successfully. These checks do not establish GPU performance
or real-data model quality.

The deployed Docker UI also completed short synthetic compression and layer
inspection runs. Focused UI server/orchestration checks passed 13 tests.
