class NixTokenizer {
    constructor(tokenizerState) {
        this.vocab_dict = tokenizerState.vocab_dict;
        this.abbreviations_dict = tokenizerState.abbreviations_dict;
        this.whitespace_regex = new RegExp(tokenizerState.whitespace_regex, 'g');

        // Create regex for abbreviations
        this.abbreviations_regex = Object.keys(this.abbreviations_dict).map(abbrev => {
            return [new RegExp(`\\b${abbrev}\\.`, 'g'), this.abbreviations_dict[abbrev]];
        });
    }

    expandAbbreviations(text) {
        for (const [regex, replacement] of this.abbreviations_regex) {
            text = text.replace(regex, replacement);
        }
        return text;
    }

    collapseWhitespace(text) {
        return text.replace(this.whitespace_regex, ' ');
    }

    intersperse(list, item) {
        const result = new Array(list.length * 2 + 1).fill(item);
        for (let i = 0; i < list.length; i++) {
            result[i * 2 + 1] = list[i];
        }
        return result;
    }

    padTokens(tokens) {
        const tokensLengths = tokens.map(token => token.length);
        const maxLen = Math.max(...tokensLengths);
        const paddedTokens = tokens.map(token => {
            const padding = new Array(maxLen - token.length).fill(0);
            return [...token, ...padding];
        });
        return [paddedTokens, tokensLengths];
    }

    async tokenize(texts) {
        // 1. Phonemize input texts
        const phonemes = [];
        for (const text of texts) {
            const expandedText = this.expandAbbreviations(text.toLowerCase());
            const phonemizedText = await window.phonemizer.phonemize(expandedText, { strip: true });
            const collapsedText = this.collapseWhitespace(phonemizedText);
            phonemes.push(collapsedText);
        }

        // 2. Tokenize phonemes
        const tokens = phonemes.map(phoneme => {
            const tokenizedPhoneme = phoneme.split('').map(p => this.vocab_dict[p]).filter(p => p);
            return this.intersperse(tokenizedPhoneme, 0);
        });

        // 3. Pad tokens
        const [paddedTokens, tokensLengths] = this.padTokens(tokens);

        return [paddedTokens, tokensLengths, phonemes];
    }
}

class NixTTS {
    constructor(modelUrls, tokenizerState) {
        this.tokenizer = new NixTokenizer(tokenizerState);
        this.encoder = null;
        this.decoder = null;
        this.modelUrls = modelUrls;
    }

    async init() {
        this.encoder = await ort.InferenceSession.create(this.modelUrls.encoder);
        this.decoder = await ort.InferenceSession.create(this.modelUrls.decoder);
    }

    async vocalize(text) {
        const [tokens, tokenLengths, phonemes] = await this.tokenizer.tokenize([text]);

        const flattenedTokens = tokens.flat();
        const c = new ort.Tensor('int64', BigInt64Array.from(flattenedTokens.map(BigInt)), [tokens.length, flattenedTokens.length / tokens.length]);
        const c_lengths = new ort.Tensor('int64', BigInt64Array.from(tokenLengths.map(BigInt)), [tokenLengths.length]);

        const encoderFeeds = { c: c, c_lengths: c_lengths };
        const encoderResults = await this.encoder.run(encoderFeeds);
        const z = encoderResults[Object.keys(encoderResults)[2]];

        const decoderFeeds = { z: z };
        const decoderResults = await this.decoder.run(decoderFeeds);
        const xw = decoderResults[Object.keys(decoderResults)[0]];

        return xw.data;
    }
}

// Placeholder for the tokenizer state. In a real application, this would be
// loaded from a JSON file.
const TOKENIZER_STATE = {
    "abbreviations_dict": {
        "mrs": "misess", "mr": "mister", "dr": "doctor", "st": "saint", "co": "company",
        "jr": "junior", "maj": "major", "gen": "general", "drs": "doctors", "rev": "reverend",
        "lt": "lieutenant", "hon": "honorable", "sgt": "sergeant", "capt": "captain",
        "esq": "esquire", "ltd": "limited", "col": "colonel", "ft": "fort"
    },
    "whitespace_regex": "\\s+",
    "vocab_dict": {
        " ": 1, "!": 2, "\"": 3, "'": 4, "(": 5, ")": 6, ",": 7, "-": 8, ".": 9, ":": 10, ";": 11,
        "?": 12, "a": 13, "b": 14, "c": 15, "d": 16, "e": 17, "f": 18, "g": 19, "h": 20, "i": 21,
        "j": 22, "k": 23, "l": 24, "m": 25, "n": 26, "o": 27, "p": 28, "q": 29, "r": 30, "s": 31,
        "t": 32, "u": 33, "v": 34, "w": 35, "x": 36, "y": 37, "z": 38, "æ": 39, "ç": 40, "ð": 41,
        "ø": 42, "ħ": 43, "ŋ": 44, "œ": 45, "ǀ": 46, "ǃ": 47, "ɐ": 48, "ɑ": 49, "ɒ": 50, "ɓ": 51,
        "ɔ": 52, "ɕ": 53, "ɖ": 54, "ɗ": 55, "ɘ": 56, "ə": 57, "ɚ": 58, "ɛ": 59, "ɜ": 60, "ɞ": 61,
        "ɟ": 62, "ɠ": 63, "ɡ": 64, "ɢ": 65, "ɣ": 66, "ɤ": 67, "ɥ": 68, "ɦ": 69, "ɧ": 70, "ɨ": 71,
        "ɩ": 72, "ɪ": 73, "ɫ": 74, "ɬ": 75, "ɭ": 76, "ɮ": 77, "ɯ": 78, "ɰ": 79, "ɱ": 80, "ɲ": 81,
        "ɳ": 82, "ɴ": 83, "ɵ": 84, "ɶ": 85, "ɸ": 86, "ɹ": 87, "ɺ": 88, "ɻ": 89, "ɽ": 90, "ɾ": 91,
        "ʀ": 92, "ʁ": 93, "ʂ": 94, "ʃ": 95, "ʄ": 96, "ʈ": 97, "ʉ": 98, "ʊ": 99, "ʋ": 100, "ʌ": 101,
        "ʍ": 102, "ʎ": 103, "ʏ": 104, "ʐ": 105, "ʑ": 106, "ʒ": 107, "ʔ": 108, "ʕ": 109, "ʘ": 110,
        "ʙ": 111, "ʛ": 112, "ʜ": 113, "ʝ": 114, "ʟ": 115, "ʡ": 116, "ʢ": 117, "əʊ": 118
    }
};

function playAudio(audioData, sampleRate = 22050) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioContext.createBuffer(1, audioData.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    channelData.set(audioData);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

async function testTokenizer() {
    const tokenizer = new NixTokenizer(TOKENIZER_STATE);
    const text = "Mr. Smith went to Washington.";
    const [tokens, tokenLengths, phonemes] = await tokenizer.tokenize([text]);
    console.log("Test Text:", text);
    console.log("Phonemes:", phonemes);
    console.log("Tokens:", tokens);
    console.log("Token Lengths:", tokenLengths);
}
