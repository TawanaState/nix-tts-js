import {phonemize} from "https://cdn.jsdelivr.net/npm/phonemizer"
export class NixTokenizer {
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
            const phonemizedText = await phonemize(expandedText);
            const collapsedText = this.collapseWhitespace(phonemizedText[0]);
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

export class NixTTS {
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

    async vocalize(text, sid) {
        // Tokenize the input text
        const [tokens, tokenLengths, phonemes] = await this.tokenizer.tokenize([text]);
        const [paddedTokens, tokensLengths] = await this.tokenizer.tokenize([text]);
        const maxTokenLength = paddedTokens[0].length;
        const flattenedTokens = paddedTokens.flat();
        const c = new ort.Tensor('int64', BigInt64Array.from(flattenedTokens.map(BigInt)), [paddedTokens.length, maxTokenLength]);
        const c_lengths = new ort.Tensor('int64', BigInt64Array.from(tokenLengths.map(BigInt)), [tokenLengths.length]);
        // Run the encoder model
        const encoderFeeds = {
            c: c,
            c_lengths: c_lengths
        };
        const encoderResults = await this.encoder.run(encoderFeeds);

        // Extract encoder outputs
        const z = encoderResults['z'];
        const d_rounded = encoderResults['duration_rounded'];

        // Create masks for the decoder input
        const y_max_length = d_rounded.dims[2];
        const y_lengths = d_rounded.data.reduce((a, b) => a + b, 0);
        const x_masks = new ort.Tensor(
            'bool',
            Array(y_lengths).fill(true),
            [1, 1, y_lengths]
        );

        // Create speaker embedding tensor
        const g = new ort.Tensor('int64', BigInt64Array.from([BigInt(sid)]));

        // Run the decoder model
        const decoderFeeds = {
            x: encoderResults['z'],
            x_masks: encoderResults['x_masks'],
            g: g,
        };
        const decoderResults = await this.decoder.run(decoderFeeds);
        const xw = decoderResults.xw;

        return xw.data;
    }
}

// Placeholder for the tokenizer state. In a real application, this would be
// loaded from a JSON file.
let tokenizer_file = await fetch("/model/tokenizer_state.json");
export let TOKENIZER_STATE = await tokenizer_file.json();

export function playAudio(audioData, sampleRate = 22050) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = audioContext.createBuffer(1, audioData.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    channelData.set(audioData);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

export async function testTokenizer() {
    const tokenizer = new NixTokenizer(TOKENIZER_STATE);
    const text = "Mr. Smith went to Washington.";
    const [tokens, tokenLengths, phonemes] = await tokenizer.tokenize([text]);
    console.log("Test Text:", text);
    console.log("Phonemes:", phonemes);
    console.log("Tokens:", tokens);
    console.log("Token Lengths:", tokenLengths);
}
// Run the tokenizer test when the page loads
        testTokenizer();