# Course language and solutions

Use the language and mode cell in `Table_of_Contents.ipynb` to select `en`, `zh`, or `tw`, and
`exercises` or `solutions`. Save open notebooks before switching, then reload
their tabs. The settings apply to the whole course. You can also run:

```sh
python composer/switch_language.py zh --mode solutions
```

Solutions supply worked implementations and finite chat inputs. Comments retain
the original input paths and describe the example cases. Changing language keeps
the selected mode. Switching modes saves your code edits and added cells for each
mode so you can return to them. Assessment code remains unsolved in every mode;
its separate answer key is not part of the language graph.

The early local-model demonstrations still require the course GPU and model
downloads. Run them in order and close completed notebook kernels before starting
6.5. Keep the 6.5 kernel running for the later notebooks. Notebook 99 retains the optional NIM deployment reference, which requires host Docker access and an NGC API key.

`translation_web.json` owns the notebook graph and code variants. The checked-in
notebooks are its English exercise projection. Locale provenance and explicit
English fallbacks accompany each translated block. Code is shared across locales.
`solutions_map.json` records each changed cell and its source digests.

The two server implementations in 6.5 are projections of `sdxl_server.py` and
`router_server.py`; tests require exact equality. `local-models.json` records the
local model/runtime compatibility contract. After editing a canonical cell,
run `python composer/generate_notebooks.py` in the Linux course environment,
then run the model and course-mode tests. `--check` detects generated drift
without writing. Regeneration is for maintainers and overwrites learner edits.
