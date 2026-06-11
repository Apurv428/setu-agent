# Recording the Setu Demo

## Run the scripted demo

The `--demo` flag runs a 3-turn scripted conversation using the scratch agent.
No microphone required.

```bash
python app.py --demo
```

What it does:

- Turn 1: asks what the most widely spoken language in India is
- Turn 2: asks how to say "good morning" in that language (follow-up using memory)
- Turn 3: asks to translate that phrase into Tamil

The agent decides which tools to call at each step. You will see each tool call
printed to the terminal, the final answer for each turn, and the path to the
saved WAV file.

## Record a terminal GIF with asciinema

Install asciinema (Linux/macOS):

```bash
pip install asciinema
```

Record:

```bash
asciinema rec demo.cast
python app.py --demo
exit
```

Convert to GIF using agg (asciinema gif generator):

```bash
# install agg: https://github.com/asciinema/agg
agg demo.cast demo.gif
```

Place `demo.gif` in the repo root. The README already references it.

## Record a terminal GIF on Windows with terminalizer

Install terminalizer:

```bash
npm install -g terminalizer
```

Record:

```bash
terminalizer record demo
python app.py --demo
# press Ctrl+D to stop recording
terminalizer render demo
```

This produces `demo.gif` in the current directory.

## Tips for a clean recording

- Set your terminal font to a monospace font at 14pt or larger.
- Use a dark theme (the output is designed for dark backgrounds).
- Keep the terminal window around 100 columns wide.
- The demo takes about 30-60 seconds to run depending on API latency.
  If you want a faster recording, edit `DEMO_TURNS` in `app.py` to use
  shorter questions.
