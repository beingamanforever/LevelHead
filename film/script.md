# LevelHead film - narration script

About 99 seconds, voiced by ElevenLabs "George" (Multilingual v2, speed 1.1, through OpenRouter).
The first 2.95 seconds are a cold open with picture, sound effects and no voice; the music enters on the impact at 2.75 seconds.
The paper is under double-blind review, so the film names no author, affiliation or venue; it links only the public repository.

| Scene | Narration |
|---|---|
| Open | Every document pipeline starts with one question: which way is up? |
| The usual order | Pipelines guess the quarter turn first, then remove the tilt. But classifiers only ever saw level pages: on this tilted page, the guess is half a turn off, and the deskewer levels it, perfectly upside down. |
| The idea | Here's the fix. Any angle is a quarter turn, plus a tilt under forty-five degrees. Rows of text reveal the tilt. Letter shapes reveal the turn. |
| Step 1 · Level | So LevelHead deskews first. Text rows draw a streak in the Fourier spectrum; we read its angle, and turn the page level. |
| Step 1 · Photos | On phone photos, the borders outshine the text. A Hann taper fades them, and the streak returns. |
| Step 2 · Ask | Then any frozen model looks at the level page four times, once per quarter turn. |
| Step 3 · Combine | Each answer is rolled back by its turn, and the four are multiplied. A model's favourite answer cancels out, and one confidently wrong view is simply outvoted. |
| Any backbone | No training. It wraps any model, from a mobile CNN to a two-hundred-billion-parameter VLM. |
| Results | Classify-first collapses at every diagonal. LevelHead stays flat, and on tilted pages every model gains, by up to sixty points. |
| Real pages | On real phone photos and scans, classifiers climb from two thirds to ninety percent. |
| Test scores | Two tests, two scores. Guessing the angle: vision-language models gain thirty-eight points on average, classifiers seven. Reading the text: OCR errors fall to the level of a perfectly upright page. |
| Close | LevelHead. Deskew before you classify. |
