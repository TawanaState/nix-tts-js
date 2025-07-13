# Nix-TTS.js: A Journey into Browser-Based TTS 🗣️

This is a JavaScript implementation of the [Nix-TTS model](https://github.com/rendchevi/nix-tts), bringing text-to-speech capabilities directly to your browser.

This project was a personal challenge and a deep dive into the world of in-browser machine learning. After a lot of wrestling with tensors, tokenizers, and phonemes, it finally works! It's a testament to what's possible on the web today with tools like ONNX Runtime.

## ✨ Features

- **Pure JavaScript:** Runs entirely in the browser. No server needed.
- **ONNX Powered:** Uses `onnxruntime-web` for efficient model inference.
- **Text-to-Phoneme:** Integrates `phonemizer` to convert input text into phonemes before tokenization.

## 🚀 How to Use

Getting started is straightforward. You'll need the ONNX models and the `tokenizer_state.json` file from the original Nix-TTS repository.

1.  **Include the necessary scripts** in your HTML file. Make sure you have `onnxruntime-web` available.

    ```html
    <!-- Or use a local copy -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script type="module" src="app.js"></script>
    ```

2.  **Use the `NixTTS` class** in your JavaScript module.

```javascript
// app.js
import { NixTTS, playAudio } from './nix-tts.js';

async function main() {
    // 1. Point to your model files and the tokenizer state
    const modelUrls = {
        encoder: '/model/nix_encoder.onnx',
        decoder: '/model/nix_decoder.onnx',
    };
    const tokenizerResponse = await fetch('/model/tokenizer_state.json');
    const tokenizerState = await tokenizerResponse.json();

    // 2. Initialize the TTS engine
    console.log("Initializing TTS...");
    const tts = new NixTTS(modelUrls, tokenizerState);
    await tts.init();
    console.log("TTS Initialized.");

    // 3. Vocalize text!
    const text = "Mr. Smith went to Washington.";
    console.log(`Vocalizing: "${text}"`);
    const audioData = await tts.vocalize(text);
    console.log("Audio generated!");

    // 4. Play the audio
    playAudio(audioData); // Default sample rate is 22050
}

main();
```

## A Note on Model Quality (The Honest Truth)

To be frank, the output quality of the Nix-TTS model itself isn't exactly state-of-the-art. While I had higher hopes, the results can be a bit robotic and sometimes garbled.

It's unclear if this is due to a nuance in my JavaScript implementation or a limitation of the original model. It was a fantastic learning experience in porting a Python ML project to the web, but if you're looking for high-quality, production-ready TTS in JavaScript, you might have better luck with other models.

### Alternative Suggestion: Kokoro-82M

For a more robust and higher-quality alternative that is known to work well in a JavaScript environment, I highly recommend checking out the **Kokoro-82M** model. It's designed for easy JS usage and generally produces much better results. You can find implementations for it that work with Transformers.js.

## 🛠️ Technical Breakdown

The library is composed of a few key parts:

-   `NixTokenizer`: This class handles all text pre-processing.
    1.  Expands common abbreviations (e.g., "Mr." -> "Mister").
    2.  Uses `phonemize` to convert the text into a sequence of phonemes.
    3.  Collapses whitespace.
    4.  Converts phonemes into integer token IDs based on the model's vocabulary.
    5.  Pads the token arrays to a uniform length for batch processing by the ONNX model.

-   `NixTTS`: The main orchestrator.
    1.  Initializes the ONNX encoder and decoder sessions.
    2.  Takes text input and uses the `NixTokenizer` to prepare it.
    3.  Feeds the tokens into the encoder model.
    4.  Feeds the encoder output into the decoder model.
    5.  Returns the raw audio waveform data.

-   `playAudio`: A simple helper function that takes the raw audio data and plays it using the Web Audio API.

---

Feel free to fork this, experiment, and see if you can improve the output!