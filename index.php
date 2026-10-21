<?php
declare(strict_types=1);

@set_time_limit(0);
@ignore_user_abort(true);
@ini_set('memory_limit', '1024M');

header('Content-Type: text/plain; charset=utf-8');
header('Cache-Control: no-store, max-age=0');

const DATASET_FILE = __DIR__ . '/dataset.json';
const VOCAB_FILE = __DIR__ . '/vocab.json';
const MERGES_FILE = __DIR__ . '/merges.txt';
const TOKENIZER_CONFIG_FILE = __DIR__ . '/tokenizer_config.json';
const TOKENIZER_FILE = __DIR__ . '/tokenizer.json';
const CONFIG_FILE = __DIR__ . '/config.json';
const MODEL_CONFIG_FILE = __DIR__ . '/model_config.json';
const WEIGHTS_FILE = __DIR__ . '/model_weights.bin';
const CHECKPOINT_FILE = __DIR__ . '/training_checkpoint.tmp';
const TRAINING_STATE_FILE = __DIR__ . '/training_state.json';
const LOCK_FILE = __DIR__ . '/training.lock';

const MODEL_MAGIC = "JBNANO1\0";
const CHECKPOINT_MAGIC = "JBNCKP1\0";

const BOS = '<|bos|>';
const EOS = '<|eos|>';
const UNK = '<|unk|>';
const PAD = '<|pad|>';
const SPECIAL_TOKENS = [PAD, BOS, EOS, UNK];

const MAX_EPOCHS = 10;
const EARLY_STOP_PATIENCE = 3;
const BASE_LEARNING_RATE = 0.001;
const MTP_LOSS_WEIGHT = 0.25;

const JSON_FLAGS =
    JSON_UNESCAPED_UNICODE |
    JSON_UNESCAPED_SLASHES |
    JSON_PRESERVE_ZERO_FRACTION;

function fail(string $message, int $status = 500): never
{
    http_response_code($status);
    echo "ERROR: {$message}\n";
    exit;
}

function readDataset(string $path): array
{
    if (!is_file($path)) {
        fail('dataset.json not found.');
    }

    $raw = file_get_contents($path);

    if (!is_string($raw) || trim($raw) === '') {
        fail('dataset.json is empty or can't access.');
    }

    try {
        $data = json_decode(
            $raw,
            true,
            512,
            JSON_THROW_ON_ERROR
        );
    } catch (Throwable $e) {
        fail('JSON bad dataset: ' . $e->getMessage());
    }

    if (isset($data['data']) && is_array($data['data'])) {
        $data = $data['data'];
    }

    if (!is_array($data)) {
        fail('dataset should be json.');
    }

    $rows = [];

    foreach ($data as $row) {
        if (!is_array($row)) {
            continue;
        }

        $input = $row['input'] ?? $row['question'] ?? null;
        $output = $row['output'] ?? $row['answer'] ?? null;

        if (!is_string($input) || !is_string($output)) {
            continue;
        }

        $input = trim($input);
        $output = trim($output);

        if ($input === '' || $output === '') {
            continue;
        }

        $rows[] = [
            'input' => $input,
            'output' => $output
        ];
    }

    if (count($rows) < 2) {
        fail('dataset is bad.');
    }

    return $rows;
}

function trainingText(array $row): string
{
    return 'User: ' . $row['input']
        . "\nAssistant: "
        . $row['output'];
}

function unicodeChars(string $text): array
{
    $parts = preg_split(
        '//u',
        $text,
        -1,
        PREG_SPLIT_NO_EMPTY
    );

    return is_array($parts) ? $parts : str_split($text);
}

function pretokenize(string $text): array
{
    $text = str_replace(
        ["\r\n", "\r"],
        "\n",
        $text
    );

    preg_match_all(
        '/(?:\s+)?[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]|\s+/u',
        $text,
        $matches
    );

    $pieces = [];

    foreach ($matches[0] ?? [] as $piece) {
        $piece = str_replace(
            [" ", "\t", "\n"],
            ['Ġ', 'ĉ', '↵'],
            $piece
        );

        $chars = unicodeChars($piece);

        if ($chars) {
            $pieces[] = $chars;
        }
    }

    if (!$pieces && $text !== '') {
        $mapped = str_replace(
            [" ", "\t", "\n"],
            ['Ġ', 'ĉ', '↵'],
            $text
        );

        foreach (unicodeChars($mapped) as $c) {
            $pieces[] = [$c];
        }
    }

    return $pieces;
}

function trainBpe(array $texts): array
{
    $pieces = [];
    $characters = [];
    $totalChars = 0;

    foreach ($texts as $text) {
        foreach (pretokenize($text) as $piece) {
            $pieces[] = $piece;
            $totalChars += count($piece);

            foreach ($piece as $character) {
                $characters[$character] = true;
            }
        }
    }

    $vocab = [];

    foreach (SPECIAL_TOKENS as $token) {
        $vocab[$token] = count($vocab);
    }

    foreach (array_keys($characters) as $character) {
        if (!array_key_exists($character, $vocab)) {
            $vocab[$character] = count($vocab);
        }
    }

    $merges = [];

    $mergeLimit = min(
        256,
        max(
            24,
            (int) ceil(sqrt(max(1, $totalChars)))
        )
    );

    for ($round = 0; $round < $mergeLimit; $round++) {
        $counts = [];
        $pairValues = [];

        foreach ($pieces as $piece) {
            $n = count($piece);

            for ($i = 0; $i < $n - 1; $i++) {
                $left = $piece[$i];
                $right = $piece[$i + 1];

                $key = $left . "\x01" . $right;

                $counts[$key] = ($counts[$key] ?? 0) + 1;

                if (!isset($pairValues[$key])) {
                    $pairValues[$key] = [$left, $right];
                }
            }
        }

        if (!$counts) {
            break;
        }

        arsort($counts, SORT_NUMERIC);

        $bestKey = array_key_first($counts);

        if ($bestKey === null || $counts[$bestKey] < 2) {
            break;
        }

        [$left, $right] = $pairValues[$bestKey];

        $joined = $left . $right;

        $merges[] = [$left, $right];

        if (!array_key_exists($joined, $vocab)) {
            $vocab[$joined] = count($vocab);
        }

        foreach ($pieces as $pi => $piece) {
            $newPiece = [];
            $i = 0;
            $n = count($piece);

            while ($i < $n) {
                if (
                    $i + 1 < $n &&
                    $piece[$i] === $left &&
                    $piece[$i + 1] === $right
                ) {
                    $newPiece[] = $joined;
                    $i += 2;
                } else {
                    $newPiece[] = $piece[$i];
                    $i++;
                }
            }

            $pieces[$pi] = $newPiece;
        }
    }

    return [
        'vocab' => $vocab,
        'merges' => $merges,
        'merge_count' => count($merges),
        'training_characters' => $totalChars
    ];
}

function saveTokenizer(array $tokenizer): void
{
    $vocabJson = json_encode(
        $tokenizer['vocab'],
        JSON_FLAGS | JSON_PRETTY_PRINT | JSON_FORCE_OBJECT
    );

    if (
        $vocabJson === false ||
        file_put_contents(
            VOCAB_FILE,
            $vocabJson . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('cant save vocab.');
    }

    $mergeLines = ['#version: 0.2'];

    foreach ($tokenizer['merges'] as [$left, $right]) {
        $mergeLines[] = $left . ' ' . $right;
    }

    if (
        file_put_contents(
            MERGES_FILE,
            implode("\n", $mergeLines) . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('cant save merges');
    }

    $config = [
        'tokenizer_class' => 'JibayBPETokenizer',
        'model_type' => 'BPE',
        'vocab_file' => 'vocab.json',
        'merges_file' => 'merges.txt',
        'tokenizer_file' => 'tokenizer.json',
        'unk_token' => UNK,
        'bos_token' => BOS,
        'eos_token' => EOS,
        'pad_token' => PAD,
        'special_tokens' => SPECIAL_TOKENS,
        'pre_tokenizer' => 'unicode-word-pieces',
        'normalizer' => [
            'space' => 'Ġ',
            'tab' => 'ĉ',
            'newline' => '↵'
        ],
        'vocab_size' => count($tokenizer['vocab']),
        'merge_count' => count($tokenizer['merges']),
        'format_version' => 1
    ];

    $cfgJson = json_encode(
        $config,
        JSON_FLAGS | JSON_PRETTY_PRINT
    );

    if (
        $cfgJson === false ||
        file_put_contents(
            TOKENIZER_CONFIG_FILE,
            $cfgJson . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('cant save tokenizer config.');
    }

    $addedTokens = [];

    foreach (SPECIAL_TOKENS as $special) {
        $addedTokens[] = [
            'id' => (int) $tokenizer['vocab'][$special],
            'special' => true,
            'content' => $special,
            'single_word' => false,
            'lstrip' => false,
            'rstrip' => false,
            'normalized' => false
        ];
    }

    $mergePairs = [];

    foreach ($tokenizer['merges'] as [$left, $right]) {
        $mergePairs[] = [$left, $right];
    }

    $full = [
        'version' => '1.0',
        'format' => 'jibay-bpe-v1',
        'normalizer' => [
            'type' => 'JibayUnicodeNormalizer',
            'space' => 'Ġ',
            'tab' => 'ĉ',
            'newline' => '↵'
        ],
        'pre_tokenizer' => [
            'type' => 'JibayUnicodeWordPieces'
        ],
        'post_processor' => null,
        'decoder' => [
            'type' => 'JibayMarkerDecoder'
        ],
        'added_tokens' => $addedTokens,
        'model' => [
            'type' => 'BPE',
            'dropout' => null,
            'unk_token' => UNK,
            'continuing_subword_prefix' => null,
            'end_of_word_suffix' => null,
            'fuse_unk' => false,
            'vocab' => $tokenizer['vocab'],
            'merges' => $mergePairs
        ]
    ];

    $fullJson = json_encode(
        $full,
        JSON_FLAGS | JSON_PRETTY_PRINT
    );

    if (
        $fullJson === false ||
        file_put_contents(
            TOKENIZER_FILE,
            $fullJson . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('cant save tokenizer');
    }
}

function loadTokenizer(): array
{
    foreach (
        [
            VOCAB_FILE,
            MERGES_FILE,
            TOKENIZER_CONFIG_FILE,
            TOKENIZER_FILE
        ] as $path
    ) {
        if (!is_file($path)) {
            fail('فایل توکنیزر ناقص است: ' . basename($path));
        }
    }

    try {
        $vocab = json_decode(
            (string) file_get_contents(VOCAB_FILE),
            true,
            512,
            JSON_THROW_ON_ERROR
        );
    } catch (Throwable $e) {
        fail('vocab.json معتبر نیست: ' . $e->getMessage());
    }

    if (
        !is_array($vocab) ||
        !isset(
            $vocab[BOS],
            $vocab[EOS],
            $vocab[UNK],
            $vocab[PAD]
        )
    ) {
        fail('توکن‌های ویژهٔ توکنیزر ناقص‌اند.');
    }

    $ids = [];

    foreach ($vocab as $token => $id) {
        $ids[(int) $id] = (string) $token;
    }

    ksort($ids);

    $lines = file(
        MERGES_FILE,
        FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES
    );

    if ($lines === false) {
        fail('خواندن merges.txt ناموفق بود.');
    }

    $merges = [];

    foreach ($lines as $line) {
        if ($line === '' || str_starts_with($line, '#')) {
            continue;
        }

        $pair = explode(' ', $line, 2);

        if (count($pair) === 2) {
            $merges[] = [$pair[0], $pair[1]];
        }
    }

    return [
        'vocab' => $vocab,
        'id_to_token' => $ids,
        'merges' => $merges
    ];
}

function tokenizerFingerprint(): string
{
    $hash = hash_init('sha256');

    foreach (
        [
            VOCAB_FILE,
            MERGES_FILE,
            TOKENIZER_CONFIG_FILE,
            TOKENIZER_FILE
        ] as $path
    ) {
        hash_update($hash, basename($path) . "\0");
        hash_update_file($hash, $path);
    }

    return hash_final($hash);
}

function bpeEncode(string $text, array $tokenizer): array
{
    $ids = [];
    $vocab = $tokenizer['vocab'];

    foreach (pretokenize($text) as $piece) {
        foreach ($tokenizer['merges'] as [$left, $right]) {
            $joined = $left . $right;
            $newPiece = [];
            $i = 0;
            $n = count($piece);

            while ($i < $n) {
                if (
                    $i + 1 < $n &&
                    $piece[$i] === $left &&
                    $piece[$i + 1] === $right
                ) {
                    $newPiece[] = $joined;
                    $i += 2;
                } else {
                    $newPiece[] = $piece[$i];
                    $i++;
                }
            }

            $piece = $newPiece;
        }

        foreach ($piece as $token) {
            $ids[] = (int) ($vocab[$token] ?? $vocab[UNK]);
        }
    }

    return $ids;
}

function tokenString(int $id, array $tokenizer): string
{
    return (string) ($tokenizer['id_to_token'][$id] ?? UNK);
}

function decodeIds(array $ids, array $tokenizer): string
{
    $text = '';

    foreach ($ids as $id) {
        $token = tokenString((int) $id, $tokenizer);

        if ($token === EOS) {
            break;
        }

        if ($token === BOS || $token === PAD) {
            continue;
        }

        if ($token === UNK) {
            $token = '�';
        }

        $text .= $token;
    }

    return trim(
        str_replace(
            ['Ġ', '↵', 'ĉ'],
            [' ', "\n", "\t"],
            $text
        )
    );
}

function percentile(array $values, float $p): int
{
    if (!$values) {
        return 1;
    }

    sort($values, SORT_NUMERIC);

    $index = (int) ceil(
        max(0.0, min(1.0, $p)) * count($values)
    ) - 1;

    $index = max(
        0,
        min(count($values) - 1, $index)
    );

    return (int) $values[$index];
}

function zeros(int $n): array
{
    return array_fill(0, $n, 0.0);
}

function zeroMatrix(int $rows, int $cols): array
{
    $matrix = [];

    for ($i = 0; $i < $rows; $i++) {
        $matrix[] = zeros($cols);
    }

    return $matrix;
}

function randomWeight(float $scale): float
{
    return (
        (mt_rand() / mt_getrandmax()) * 2.0 - 1.0
    ) * $scale;
}

function randomMatrix(int $rows, int $cols): array
{
    $scale = sqrt(6.0 / max(1, $rows + $cols));
    $matrix = [];

    for ($i = 0; $i < $rows; $i++) {
        $row = [];

        for ($j = 0; $j < $cols; $j++) {
            $row[] = randomWeight($scale);
        }

        $matrix[] = $row;
    }

    return $matrix;
}

function deriveModelConfig(
    array $rows,
    array $tokenizer,
    string $datasetHash,
    string $tokenizerHash
): array {
    $sequenceLengths = [];
    $answerLengths = [];
    $totalTokens = 0;

    $bosId = (int) $tokenizer['vocab'][BOS];
    $eosId = (int) $tokenizer['vocab'][EOS];

    foreach ($rows as $row) {
        $sequence = array_merge(
            [$bosId],
            bpeEncode(trainingText($row), $tokenizer),
            [$eosId]
        );

        $answerLength = count(
            bpeEncode($row['output'], $tokenizer)
        );

        $sequenceLengths[] = count($sequence);
        $answerLengths[] = max(1, $answerLength);
        $totalTokens += count($sequence);
    }

    $p90Sequence = percentile($sequenceLengths, 0.90);

    $computeContext = max(
        16,
        (int) ceil(
            pow(max(1, $totalTokens), 1.0 / 3.0) * 3.0
        )
    );

    $context = max(
        8,
        min($p90Sequence, $computeContext)
    );

    $dModel = count($rows) >= 256 ? 16 : 8;
    $heads = $dModel === 8 ? 2 : 4;
    $layers = count($rows) >= 2000 ? 3 : 2;
    $ffDim = $dModel * 2;

    $maxAnswer = max(
        1,
        percentile($answerLengths, 0.90)
    );

    $generationLimit = max(
        1,
        (int) ceil($maxAnswer * 1.5)
    );

    return [
        'architectures' => ['JibayNanoForCausalLM'],
        'model_type' => 'jibay_nano',
        'model_name' => 'Jibay Nano',
        'format_version' => 1,
        'dataset_sha256' => $datasetHash,
        'tokenizer_sha256' => $tokenizerHash,
        'vocab_size' => count($tokenizer['vocab']),
        'd_model' => $dModel,
        'n_heads' => $heads,
        'n_layers' => $layers,
        'ff_dim' => $ffDim,
        'max_position_embeddings' => $context,
        'max_context_source' => '90th_percentile_training_sequence_length_and_corpus_token_count',
        'generation_limit' => $generationLimit,
        'generation_limit_source' => '90th_percentile_training_answer_length',
        'trained_examples' => count($rows),
        'training_epochs' => 0,
        'best_epoch' => 0,
        'training_loss' => null,
        'batch_size_last_training' => null,
        'optimizer' => 'Adam',
        'attention' => 'causal_multi_head_self_attention',
        'normalization' => 'pre_layer_norm',
        'activation' => 'GELU',
        'mtp' => 'auxiliary_two_token_ahead_head_with_rejection_sampling',
        'parameters' => 0
    ];
}

function identityConfig(array $cfg): array
{
    $keys = [
        'architectures',
        'model_type',
        'format_version',
        'dataset_sha256',
        'tokenizer_sha256',
        'vocab_size',
        'd_model',
        'n_heads',
        'n_layers',
        'ff_dim',
        'max_position_embeddings',
        'generation_limit',
        'attention',
        'normalization',
        'activation',
        'mtp'
    ];

    $identity = [];

    foreach ($keys as $key) {
        $identity[$key] = $cfg[$key] ?? null;
    }

    return $identity;
}

function identityHash(array $cfg): string
{
    return hash(
        'sha256',
        json_encode(identityConfig($cfg), JSON_FLAGS)
    );
}

function saveConfig(array $cfg): void
{
    $json = json_encode(
        $cfg,
        JSON_FLAGS | JSON_PRETTY_PRINT
    );

    if (
        $json === false ||
        file_put_contents(
            CONFIG_FILE,
            $json . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('ذخیرهٔ config.json ناموفق بود.');
    }

    if (
        file_put_contents(
            MODEL_CONFIG_FILE,
            $json . "\n",
            LOCK_EX
        ) === false
    ) {
        fail('ذخیرهٔ model_config.json ناموفق بود.');
    }
}

function readConfig(): ?array
{
    if (!is_file(CONFIG_FILE)) {
        return null;
    }

    try {
        $cfg = json_decode(
            (string) file_get_contents(CONFIG_FILE),
            true,
            512,
            JSON_THROW_ON_ERROR
        );
    } catch (Throwable $e) {
        fail('config.json نامعتبر است: ' . $e->getMessage());
    }

    return is_array($cfg) ? $cfg : null;
}

function initParams(array $cfg): array
{
    $vocab = (int) $cfg['vocab_size'];
    $d = (int) $cfg['d_model'];
    $ff = (int) $cfg['ff_dim'];
    $context = (int) $cfg['max_position_embeddings'];

    $params = [
        'tok_emb' => [],
        'pos_emb' => [],
        'blocks' => [],
        'final_g' => array_fill(0, $d, 1.0),
        'final_b' => zeros($d),
        'Wout' => randomMatrix($d, $vocab),
        'bout' => zeros($vocab),
        'Wmtp' => randomMatrix($d, $vocab),
        'bmtp' => zeros($vocab)
    ];

    for ($i = 0; $i < $vocab; $i++) {
        $row = [];

        for ($j = 0; $j < $d; $j++) {
            $row[] = randomWeight(0.04);
        }

        $params['tok_emb'][] = $row;
    }

    for ($i = 0; $i < $context; $i++) {
        $row = [];

        for ($j = 0; $j < $d; $j++) {
            $row[] = randomWeight(0.02);
        }

        $params['pos_emb'][] = $row;
    }

    for ($l = 0; $l < (int) $cfg['n_layers']; $l++) {
        $params['blocks'][] = [
            'ln1_g' => array_fill(0, $d, 1.0),
            'ln1_b' => zeros($d),
            'Wq' => randomMatrix($d, $d),
            'Wk' => randomMatrix($d, $d),
            'Wv' => randomMatrix($d, $d),
            'Wo' => randomMatrix($d, $d),
            'bo' => zeros($d),
            'ln2_g' => array_fill(0, $d, 1.0),
            'ln2_b' => zeros($d),
            'W1' => randomMatrix($d, $ff),
            'b1' => zeros($ff),
            'W2' => randomMatrix($ff, $d),
            'b2' => zeros($d)
        ];
    }

    return $params;
}

function zeroLike(mixed $value): mixed
{
    if (!is_array($value)) {
        return 0.0;
    }

    $out = [];

    foreach ($value as $key => $child) {
        $out[$key] = zeroLike($child);
    }

    return $out;
}

function addNested(array &$target, array $source): void
{
    foreach ($target as $key => &$value) {
        if (is_array($value)) {
            addNested($value, $source[$key]);
        } else {
            $value += (float) $source[$key];
        }
    }

    unset($value);
}

function scaleNested(array &$value, float $scale): void
{
    foreach ($value as &$child) {
        if (is_array($child)) {
            scaleNested($child, $scale);
        } else {
            $child *= $scale;
        }
    }

    unset($child);
}

function sumSquares(mixed $value): float
{
    if (!is_array($value)) {
        return (float) $value * (float) $value;
    }

    $sum = 0.0;

    foreach ($value as $child) {
        $sum += sumSquares($child);
    }

    return $sum;
}

function adamUpdate(
    array &$params,
    array $grads,
    array &$m,
    array &$v,
    int $step,
    float $lr
): void {
    $beta1 = 0.9;
    $beta2 = 0.999;
    $epsilon = 1.0e-8;

    $correction1 = 1.0 - pow($beta1, $step);
    $correction2 = 1.0 - pow($beta2, $step);

    foreach ($params as $key => &$parameter) {
        if (is_array($parameter)) {
            adamUpdate(
                $parameter,
                $grads[$key],
                $m[$key],
                $v[$key],
                $step,
                $lr
            );
        } else {
            $g = (float) $grads[$key];

            $m[$key] =
                $beta1 * (float) $m[$key] +
                (1.0 - $beta1) * $g;

            $v[$key] =
                $beta2 * (float) $v[$key] +
                (1.0 - $beta2) * $g * $g;

            $mHat = $m[$key] / $correction1;
            $vHat = $v[$key] / $correction2;

            $parameter -=
                $lr * $mHat / (sqrt($vHat) + $epsilon);
        }
    }

    unset($parameter);
}

function flattenNumbers(array $value, array &$out): void
{
    foreach ($value as $item) {
        if (is_array($item)) {
            flattenNumbers($item, $out);
        } else {
            $out[] = (float) $item;
        }
    }
}

function assignFlat(array &$target, array $flat, int &$index): void
{
    foreach ($target as &$item) {
        if (is_array($item)) {
            assignFlat($item, $flat, $index);
        } else {
            if (!array_key_exists($index, $flat)) {
                fail('تعداد وزن‌های فایل با معماری سازگار نیست.');
            }

            $item = (float) $flat[$index++];
        }
    }

    unset($item);
}

function parameterCount(array $params): int
{
    $flat = [];
    flattenNumbers($params, $flat);
    return count($flat);
}

function packFloatVector(array $values): string
{
    $data = '';
    $n = count($values);

    for ($i = 0; $i < $n; $i += 2048) {
        $data .= pack(
            'g*',
            ...array_slice($values, $i, 2048)
        );
    }

    return $data;
}

function unpackFloatVector(string $data, int $count): array
{
    if (strlen($data) !== $count * 4) {
        fail('حجم دادهٔ باینری وزن‌ها نامعتبر است.');
    }

    $values = [];
    $offset = 0;
    $length = strlen($data);

    while ($offset < $length) {
        $take = min(
            2048,
            intdiv($length - $offset, 4)
        );

        $part = unpack(
            'g*',
            substr($data, $offset, $take * 4)
        );

        if (!is_array($part)) {
            fail('خواندن وزن‌های باینری ناموفق بود.');
        }

        foreach ($part as $value) {
            $values[] = (float) $value;
        }

        $offset += $take * 4;
    }

    return $values;
}

function saveWeights(array $params, array $cfg): void
{
    $flat = [];
    flattenNumbers($params, $flat);

    $blob =
        MODEL_MAGIC .
        hex2bin(identityHash($cfg)) .
        packFloatVector($flat);

    $tmp = WEIGHTS_FILE . '.tmp.' . getmypid();

    if (
        file_put_contents($tmp, $blob, LOCK_EX) === false ||
        !rename($tmp, WEIGHTS_FILE)
    ) {
        @unlink($tmp);
        fail('ذخیرهٔ model_weights.bin ناموفق بود.');
    }
}

function loadWeights(array &$params, array $cfg): bool
{
    if (!is_file(WEIGHTS_FILE)) {
        return false;
    }

    $handle = fopen(WEIGHTS_FILE, 'rb');

    if ($handle === false) {
        return false;
    }

    $magic = fread($handle, 8);
    $hash = fread($handle, 32);

    if (
        $magic !== MODEL_MAGIC ||
        strlen($hash) !== 32 ||
        !hash_equals(
            hex2bin(identityHash($cfg)),
            $hash
        )
    ) {
        fclose($handle);
        return false;
    }

    $blob = stream_get_contents($handle);
    fclose($handle);

    if (!is_string($blob)) {
        return false;
    }

    $count = parameterCount($params);

    if (strlen($blob) !== $count * 4) {
        return false;
    }

    $flat = unpackFloatVector($blob, $count);
    $index = 0;

    assignFlat($params, $flat, $index);

    return true;
}

function saveCheckpoint(
    array $state,
    array $params,
    array $m,
    array $v,
    array $bestParams,
    array $cfg
): void {
    $meta = $state;
    $meta['identity_sha256'] = identityHash($cfg);

    $meta['parameter_count'] =
        (int) ($state['parameters'] ?? 0);

    if ($meta['parameter_count'] < 1) {
        $meta['parameter_count'] = parameterCount($params);
    }

    $meta['saved_at'] = date(DATE_ATOM);

    $json = json_encode(
        $meta,
        JSON_FLAGS | JSON_PARTIAL_OUTPUT_ON_ERROR
    );

    if ($json === false) {
        fail('ساخت فرادادهٔ checkpoint ناموفق بود.');
    }

    $vectors = [];

    foreach ([$params, $m, $v, $bestParams] as $tree) {
        $flat = [];
        flattenNumbers($tree, $flat);
        $vectors[] = packFloatVector($flat);
    }

    $blob =
        CHECKPOINT_MAGIC .
        hex2bin(identityHash($cfg)) .
        pack('V', strlen($json)) .
        $json .
        implode('', $vectors);

    $tmp = CHECKPOINT_FILE . '.writing.' . getmypid();

    if (
        file_put_contents($tmp, $blob, LOCK_EX) === false ||
        !rename($tmp, CHECKPOINT_FILE)
    ) {
        @unlink($tmp);
        fail('ذخیرهٔ checkpoint ناموفق بود.');
    }

    $publicState = $state;
    unset($publicState['order']);

    $publicJson = json_encode(
        $publicState,
        JSON_FLAGS | JSON_PRETTY_PRINT | JSON_PARTIAL_OUTPUT_ON_ERROR
    );

    if ($publicJson !== false) {
        @file_put_contents(
            TRAINING_STATE_FILE,
            $publicJson . "\n",
            LOCK_EX
        );
    }
}

function loadCheckpoint(
    array &$params,
    array &$m,
    array &$v,
    array &$bestParams,
    array $cfg
): ?array {
    if (!is_file(CHECKPOINT_FILE)) {
        return null;
    }

    $fp = fopen(CHECKPOINT_FILE, 'rb');

    if ($fp === false) {
        return null;
    }

    $magic = fread($fp, 8);
    $hash = fread($fp, 32);
    $lenRaw = fread($fp, 4);

    if (
        $magic !== CHECKPOINT_MAGIC ||
        strlen($hash) !== 32 ||
        strlen($lenRaw) !== 4 ||
        !hash_equals(
            hex2bin(identityHash($cfg)),
            $hash
        )
    ) {
        fclose($fp);
        return null;
    }

    $lenData = unpack('Vlength', $lenRaw);

    $metaRaw = fread(
        $fp,
        (int) $lenData['length']
    );

    if (
        !is_string($metaRaw) ||
        strlen($metaRaw) !== (int) $lenData['length']
    ) {
        fclose($fp);
        return null;
    }

    try {
        $state = json_decode(
            $metaRaw,
            true,
            512,
            JSON_THROW_ON_ERROR
        );
    } catch (Throwable $e) {
        fclose($fp);
        return null;
    }

    if (
        !is_array($state) ||
        ($state['identity_sha256'] ?? '') !== identityHash($cfg)
    ) {
        fclose($fp);
        return null;
    }

    $count = parameterCount($params);

    if (($state['parameter_count'] ?? 0) !== $count) {
        fclose($fp);
        return null;
    }

    $remaining = stream_get_contents($fp);
    fclose($fp);

    if (
        !is_string($remaining) ||
        strlen($remaining) !== $count * 4 * 4
    ) {
        return null;
    }

    $offset = 0;

    foreach (['params', 'm', 'v', 'best'] as $which) {
        $part = substr(
            $remaining,
            $offset,
            $count * 4
        );

        $offset += $count * 4;

        $flat = unpackFloatVector($part, $count);
        $index = 0;

        if ($which === 'params') {
            assignFlat($params, $flat, $index);
        } elseif ($which === 'm') {
            assignFlat($m, $flat, $index);
        } elseif ($which === 'v') {
            assignFlat($v, $flat, $index);
        } else {
            assignFlat($bestParams, $flat, $index);
        }

        unset($flat);
    }

    return $state;
}

function layerNorm(
    array $x,
    array $gamma,
    array $beta
): array {
    $d = count($x);
    $mean = array_sum($x) / max(1, $d);
    $variance = 0.0;

    foreach ($x as $value) {
        $variance += ($value - $mean) * ($value - $mean);
    }

    $variance /= max(1, $d);
    $inv = 1.0 / sqrt($variance + 1.0e-5);

    $xhat = [];
    $y = [];

    for ($i = 0; $i < $d; $i++) {
        $xhat[$i] = ($x[$i] - $mean) * $inv;
        $y[$i] = $xhat[$i] * $gamma[$i] + $beta[$i];
    }

    return [
        'y' => $y,
        'xhat' => $xhat,
        'inv' => $inv
    ];
}

function layerNormBackward(
    array $xhat,
    float $inv,
    array $gamma,
    array $dy,
    array &$dg,
    array &$db
): array {
    $d = count($dy);
    $dxhat = [];
    $sum1 = 0.0;
    $sum2 = 0.0;

    for ($i = 0; $i < $d; $i++) {
        $dg[$i] += $dy[$i] * $xhat[$i];
        $db[$i] += $dy[$i];
        $dxhat[$i] = $dy[$i] * $gamma[$i];

        $sum1 += $dxhat[$i];
        $sum2 += $dxhat[$i] * $xhat[$i];
    }

    $dx = [];

    for ($i = 0; $i < $d; $i++) {
        $dx[$i] =
            ($inv / max(1, $d)) *
            (
                $d * $dxhat[$i] -
                $sum1 -
                $xhat[$i] * $sum2
            );
    }

    return $dx;
}

function linear(
    array $x,
    array $w,
    ?array $bias = null
): array {
    $in = count($x);
    $out = count($w[0]);
    $y = $bias ?? zeros($out);

    for ($i = 0; $i < $in; $i++) {
        $xi = $x[$i];

        if ($xi == 0.0) {
            continue;
        }

        for ($j = 0; $j < $out; $j++) {
            $y[$j] += $xi * $w[$i][$j];
        }
    }

    return $y;
}

function gelu(float $x): float
{
    return 0.5 * $x * (
        1.0 + tanh(
            0.7978845608 *
            ($x + 0.044715 * $x * $x * $x)
        )
    );
}

function geluDerivative(float $x): float
{
    $a = 0.7978845608;

    $u = $a * (
        $x + 0.044715 * $x * $x * $x
    );

    $t = tanh($u);

    $du = $a * (
        1.0 + 3.0 * 0.044715 * $x * $x
    );

    return
        0.5 * (1.0 + $t) +
        0.5 * $x * (1.0 - $t * $t) * $du;
}

function softmax(array $logits): array
{
    if (!$logits) {
        return [];
    }

    $max = max($logits);
    $exps = [];
    $sum = 0.0;

    foreach ($logits as $i => $value) {
        $exp = exp(
            max(-80.0, min(0.0, (float) $value - $max))
        );

        $exps[$i] = $exp;
        $sum += $exp;
    }

    if ($sum <= 0.0) {
        return array_fill(
            0,
            count($logits),
            1.0 / max(1, count($logits))
        );
    }

    foreach ($exps as &$value) {
        $value /= $sum;
    }

    unset($value);

    return $exps;
}

function forwardTransformer(
    array $ids,
    array $params,
    array $cfg
): array {
    $contextLimit = (int) $cfg['max_position_embeddings'];

    if (count($ids) > $contextLimit) {
        $ids = array_slice($ids, -$contextLimit);
    }

    $n = count($ids);
    $d = (int) $cfg['d_model'];
    $heads = (int) $cfg['n_heads'];
    $headDim = intdiv($d, $heads);

    $x = [];

    for ($t = 0; $t < $n; $t++) {
        $id = (int) $ids[$t];
        $row = [];

        for ($j = 0; $j < $d; $j++) {
            $row[$j] =
                $params['tok_emb'][$id][$j] +
                $params['pos_emb'][$t][$j];
        }

        $x[] = $row;
    }

    $blockCaches = [];

    foreach ($params['blocks'] as $layerIndex => $p) {
        $input = $x;

        $norm1 = [];
        $ln1Cache = [];
        $q = [];
        $k = [];
        $v = [];

        for ($t = 0; $t < $n; $t++) {
            $ln = layerNorm(
                $input[$t],
                $p['ln1_g'],
                $p['ln1_b']
            );

            $norm1[$t] = $ln['y'];

            $ln1Cache[$t] = [
                'xhat' => $ln['xhat'],
                'inv' => $ln['inv']
            ];

            $q[$t] = linear($norm1[$t], $p['Wq']);
            $k[$t] = linear($norm1[$t], $p['Wk']);
            $v[$t] = linear($norm1[$t], $p['Wv']);
        }

        $prob = [];
        $ctx = zeroMatrix($n, $d);
        $scale = 1.0 / sqrt(max(1, $headDim));

        for ($t = 0; $t < $n; $t++) {
            $prob[$t] = [];

            for ($head = 0; $head < $heads; $head++) {
                $base = $head * $headDim;
                $scores = [];

                for ($s = 0; $s <= $t; $s++) {
                    $dot = 0.0;

                    for ($j = 0; $j < $headDim; $j++) {
                        $dot +=
                            $q[$t][$base + $j] *
                            $k[$s][$base + $j];
                    }

                    $scores[$s] = $dot * $scale;
                }

                $prob[$t][$head] = softmax($scores);

                for ($s = 0; $s <= $t; $s++) {
                    $ps = $prob[$t][$head][$s];

                    for ($j = 0; $j < $headDim; $j++) {
                        $ctx[$t][$base + $j] +=
                            $ps * $v[$s][$base + $j];
                    }
                }
            }
        }

        $res1 = [];
        $norm2 = [];
        $ln2Cache = [];
        $u = [];
        $act = [];
        $out = [];

        for ($t = 0; $t < $n; $t++) {
            $attout = linear(
                $ctx[$t],
                $p['Wo'],
                $p['bo']
            );

            $res1[$t] = [];

            for ($j = 0; $j < $d; $j++) {
                $res1[$t][$j] =
                    $input[$t][$j] + $attout[$j];
            }

            $ln = layerNorm(
                $res1[$t],
                $p['ln2_g'],
                $p['ln2_b']
            );

            $norm2[$t] = $ln['y'];

            $ln2Cache[$t] = [
                'xhat' => $ln['xhat'],
                'inv' => $ln['inv']
            ];

            $u[$t] = linear(
                $norm2[$t],
                $p['W1'],
                $p['b1']
            );

            $act[$t] = [];

            foreach ($u[$t] as $j => $value) {
                $act[$t][$j] = gelu((float) $value);
            }

            $ffout = linear(
                $act[$t],
                $p['W2'],
                $p['b2']
            );

            $out[$t] = [];

            for ($j = 0; $j < $d; $j++) {
                $out[$t][$j] =
                    $res1[$t][$j] + $ffout[$j];
            }
        }

        $blockCaches[] = [
            'input' => $input,
            'norm1' => $norm1,
            'ln1' => $ln1Cache,
            'q' => $q,
            'k' => $k,
            'v' => $v,
            'prob' => $prob,
            'ctx' => $ctx,
            'res1' => $res1,
            'norm2' => $norm2,
            'ln2' => $ln2Cache,
            'u' => $u,
            'act' => $act,
            'output' => $out
        ];

        $x = $out;
    }

    $finalNorm = [];
    $finalCache = [];
    $logits = [];
    $mtpLogits = [];

    for ($t = 0; $t < $n; $t++) {
        $ln = layerNorm(
            $x[$t],
            $params['final_g'],
            $params['final_b']
        );

        $finalNorm[$t] = $ln['y'];

        $finalCache[$t] = [
            'xhat' => $ln['xhat'],
            'inv' => $ln['inv']
        ];

        $logits[$t] = linear(
            $finalNorm[$t],
            $params['Wout'],
            $params['bout']
        );

        $mtpLogits[$t] = linear(
            $finalNorm[$t],
            $params['Wmtp'],
            $params['bmtp']
        );
    }

    return [
        'ids' => $ids,
        'logits' => $logits,
        'mtp_logits' => $mtpLogits,
        'final_norm' => $finalNorm,
        'final_cache' => $finalCache,
        'blocks' => $blockCaches
    ];
}

function backwardTransformer(
    array $fwd,
    array $params,
    array &$grads,
    array $cfg,
    float $mtpWeight = MTP_LOSS_WEIGHT
): float {
    $n = count($fwd['ids']);
    $d = count($params['final_g']);
    $vocabSize = count($params['bout']);

    if ($n < 2) {
        return 0.0;
    }

    $mainPositions = max(1, $n - 1);
    $mtpPositions = max(1, $n - 2);

    $dFinalNorm = zeroMatrix($n, $d);
    $loss = 0.0;

    for ($t = 0; $t < $n - 1; $t++) {
        $target = (int) $fwd['ids'][$t + 1];
        $probs = softmax($fwd['logits'][$t]);

        $loss -=
            log(max(1.0e-12, (float) $probs[$target])) /
            $mainPositions;

        for ($token = 0; $token < $vocabSize; $token++) {
            $g = (
                $probs[$token] -
                ($token === $target ? 1.0 : 0.0)
            ) / $mainPositions;

            $grads['bout'][$token] += $g;

            for ($j = 0; $j < $d; $j++) {
                $grads['Wout'][$j][$token] +=
                    $fwd['final_norm'][$t][$j] * $g;

                $dFinalNorm[$t][$j] +=
                    $params['Wout'][$j][$token] * $g;
            }
        }

        if ($t + 2 < $n) {
            $target2 = (int) $fwd['ids'][$t + 2];

            $probs2 = softmax(
                $fwd['mtp_logits'][$t]
            );

            $loss -=
                $mtpWeight *
                log(max(1.0e-12, (float) $probs2[$target2])) /
                $mtpPositions;

            for ($token = 0; $token < $vocabSize; $token++) {
                $g2 = (
                    $probs2[$token] -
                    ($token === $target2 ? 1.0 : 0.0)
                ) * $mtpWeight / $mtpPositions;

                $grads['bmtp'][$token] += $g2;

                for ($j = 0; $j < $d; $j++) {
                    $grads['Wmtp'][$j][$token] +=
                        $fwd['final_norm'][$t][$j] * $g2;

                    $dFinalNorm[$t][$j] +=
                        $params['Wmtp'][$j][$token] * $g2;
                }
            }
        }
    }

    $dX = zeroMatrix($n, $d);

    for ($t = 0; $t < $n; $t++) {
        $dX[$t] = layerNormBackward(
            $fwd['final_cache'][$t]['xhat'],
            $fwd['final_cache'][$t]['inv'],
            $params['final_g'],
            $dFinalNorm[$t],
            $grads['final_g'],
            $grads['final_b']
        );
    }

    $heads = (int) $cfg['n_heads'];
    $headDim = intdiv($d, $heads);

    for (
        $layerIndex = count($params['blocks']) - 1;
        $layerIndex >= 0;
        $layerIndex--
    ) {
        $p = $params['blocks'][$layerIndex];
        $c = $fwd['blocks'][$layerIndex];

        $dRes1 = $dX;
        $dNorm2 = zeroMatrix($n, $d);
        $ff = count($p['b1']);

        for ($t = 0; $t < $n; $t++) {
            $dAct = zeros($ff);

            for ($i = 0; $i < $ff; $i++) {
                for ($j = 0; $j < $d; $j++) {
                    $g = $dX[$t][$j];

                    $grads['blocks'][$layerIndex]['W2'][$i][$j] +=
                        $c['act'][$t][$i] * $g;

                    $grads['blocks'][$layerIndex]['b2'][$j] += $g;

                    $dAct[$i] +=
                        $p['W2'][$i][$j] * $g;
                }
            }

            $dU = zeros($ff);

            for ($i = 0; $i < $ff; $i++) {
                $dU[$i] =
                    $dAct[$i] *
                    geluDerivative((float) $c['u'][$t][$i]);
            }

            for ($i = 0; $i < $d; $i++) {
                for ($j = 0; $j < $ff; $j++) {
                    $grads['blocks'][$layerIndex]['W1'][$i][$j] +=
                        $c['norm2'][$t][$i] * $dU[$j];

                    $dNorm2[$t][$i] +=
                        $p['W1'][$i][$j] * $dU[$j];
                }
            }

            for ($j = 0; $j < $ff; $j++) {
                $grads['blocks'][$layerIndex]['b1'][$j] +=
                    $dU[$j];
            }
        }

        for ($t = 0; $t < $n; $t++) {
            $dFromLn2 = layerNormBackward(
                $c['ln2'][$t]['xhat'],
                $c['ln2'][$t]['inv'],
                $p['ln2_g'],
                $dNorm2[$t],
                $grads['blocks'][$layerIndex]['ln2_g'],
                $grads['blocks'][$layerIndex]['ln2_b']
            );

            for ($j = 0; $j < $d; $j++) {
                $dRes1[$t][$j] += $dFromLn2[$j];
            }
        }

        $dInputDirect = $dRes1;
        $dCtx = zeroMatrix($n, $d);

        for ($t = 0; $t < $n; $t++) {
            for ($i = 0; $i < $d; $i++) {
                for ($j = 0; $j < $d; $j++) {
                    $grads['blocks'][$layerIndex]['Wo'][$i][$j] +=
                        $c['ctx'][$t][$i] * $dRes1[$t][$j];

                    $dCtx[$t][$i] +=
                        $p['Wo'][$i][$j] * $dRes1[$t][$j];
                }
            }

            for ($j = 0; $j < $d; $j++) {
                $grads['blocks'][$layerIndex]['bo'][$j] +=
                    $dRes1[$t][$j];
            }
        }

        $scale = 1.0 / sqrt(max(1, $headDim));

        $dq = zeroMatrix($n, $d);
        $dk = zeroMatrix($n, $d);
        $dv = zeroMatrix($n, $d);

        for ($t = 0; $t < $n; $t++) {
            for ($head = 0; $head < $heads; $head++) {
                $base = $head * $headDim;
                $dp = [];
                $weighted = 0.0;

                for ($s = 0; $s <= $t; $s++) {
                    $dot = 0.0;

                    for ($j = 0; $j < $headDim; $j++) {
                        $dot +=
                            $dCtx[$t][$base + $j] *
                            $c['v'][$s][$base + $j];
                    }

                    $dp[$s] = $dot;

                    $weighted +=
                        $c['prob'][$t][$head][$s] * $dot;
                }

                for ($s = 0; $s <= $t; $s++) {
                    $prob = $c['prob'][$t][$head][$s];

                    $ds = $prob * (
                        $dp[$s] - $weighted
                    );

                    for ($j = 0; $j < $headDim; $j++) {
                        $dq[$t][$base + $j] +=
                            $ds *
                            $c['k'][$s][$base + $j] *
                            $scale;

                        $dk[$s][$base + $j] +=
                            $ds *
                            $c['q'][$t][$base + $j] *
                            $scale;

                        $dv[$s][$base + $j] +=
                            $prob * $dCtx[$t][$base + $j];
                    }
                }
            }
        }

        $dNorm1 = zeroMatrix($n, $d);

        for ($t = 0; $t < $n; $t++) {
            for ($i = 0; $i < $d; $i++) {
                for ($j = 0; $j < $d; $j++) {
                    $grads['blocks'][$layerIndex]['Wq'][$i][$j] +=
                        $c['norm1'][$t][$i] * $dq[$t][$j];

                    $grads['blocks'][$layerIndex]['Wk'][$i][$j] +=
                        $c['norm1'][$t][$i] * $dk[$t][$j];

                    $grads['blocks'][$layerIndex]['Wv'][$i][$j] +=
                        $c['norm1'][$t][$i] * $dv[$t][$j];

                    $dNorm1[$t][$i] +=
                        $p['Wq'][$i][$j] * $dq[$t][$j] +
                        $p['Wk'][$i][$j] * $dk[$t][$j] +
                        $p['Wv'][$i][$j] * $dv[$t][$j];
                }
            }
        }

        $nextDX = zeroMatrix($n, $d);

        for ($t = 0; $t < $n; $t++) {
            $dFromLn1 = layerNormBackward(
                $c['ln1'][$t]['xhat'],
                $c['ln1'][$t]['inv'],
                $p['ln1_g'],
                $dNorm1[$t],
                $grads['blocks'][$layerIndex]['ln1_g'],
                $grads['blocks'][$layerIndex]['ln1_b']
            );

            for ($j = 0; $j < $d; $j++) {
                $nextDX[$t][$j] =
                    $dInputDirect[$t][$j] +
                    $dFromLn1[$j];
            }
        }

        $dX = $nextDX;
    }

    foreach ($fwd['ids'] as $t => $id) {
        $id = (int) $id;

        for ($j = 0; $j < $d; $j++) {
            $grads['tok_emb'][$id][$j] += $dX[$t][$j];
            $grads['pos_emb'][$t][$j] += $dX[$t][$j];
        }
    }

    return $loss;
}

function buildTrainingSequences(
    array $rows,
    array $tokenizer
): array {
    $sequences = [];
    $answerLengths = [];
    $sequenceLengths = [];

    $bosId = (int) $tokenizer['vocab'][BOS];
    $eosId = (int) $tokenizer['vocab'][EOS];

    foreach ($rows as $row) {
        $ids = array_merge(
            [$bosId],
            bpeEncode(trainingText($row), $tokenizer),
            [$eosId]
        );

        $sequences[] = $ids;
        $sequenceLengths[] = count($ids);

        $answerLengths[] = max(
            1,
            count(bpeEncode($row['output'], $tokenizer))
        );
    }

    return [
        'sequences' => $sequences,
        'sequence_lengths' => $sequenceLengths,
        'answer_lengths' => $answerLengths
    ];
}

function initialTrainingState(
    array $sequences,
    int $batchSize,
    array $params,
    array $cfg,
    bool $fineTune
): array {
    $priorLoss =
        $fineTune && is_numeric($cfg['training_loss'] ?? null)
            ? (float) $cfg['training_loss']
            : null;

    return [
        'epoch' => 1,
        'order' => [],
        'batch_offset' => 0,
        'global_step' => 0,
        'current_epoch_loss' => 0.0,
        'current_epoch_batches' => 0,
        'best_loss' => $priorLoss,
        'best_epoch' => $fineTune
            ? (int) ($cfg['best_epoch'] ?? 0)
            : 0,
        'stale_epochs' => 0,
        'epoch_losses' => [],
        'batch_size' => $batchSize,
        'finished' => false,
        'started_at' => date(DATE_ATOM),
        'examples' => count($sequences),
        'parameters' => parameterCount($params)
    ];
}

function trainModel(
    array $sequences,
    array &$params,
    array &$cfg,
    int $requestedBatchSize,
    bool $fineTune
): array {
    $m = zeroLike($params);
    $v = zeroLike($params);
    $bestParams = $params;

    $state = null;
    $resumed = false;

    $checkpointLoaded = loadCheckpoint(
        $params,
        $m,
        $v,
        $bestParams,
        $cfg
    );

    if (
        is_array($checkpointLoaded) &&
        !empty($checkpointLoaded['finished'])
    ) {
        $params = $bestParams;

        $cfg['training_epochs'] = count(
            $checkpointLoaded['epoch_losses'] ?? []
        );

        $cfg['best_epoch'] =
            (int) ($checkpointLoaded['best_epoch'] ?? 0);

        $cfg['training_loss'] =
            is_numeric($checkpointLoaded['best_loss'] ?? null)
                ? (float) $checkpointLoaded['best_loss']
                : null;

        $cfg['batch_size_last_training'] = max(
            1,
            (int) (
                $checkpointLoaded['batch_size'] ??
                $requestedBatchSize
            )
        );

        $cfg['training_steps'] =
            (int) ($checkpointLoaded['global_step'] ?? 0);

        $cfg['epoch_losses'] =
            $checkpointLoaded['epoch_losses'] ?? [];

        $cfg['parameters'] = parameterCount($params);

        saveWeights($params, $cfg);
        saveConfig($cfg);

        @unlink(CHECKPOINT_FILE);

        return [
            'params' => $params,
            'config' => $cfg,
            'resumed' => true,
            'progress' => [],
            'state' => $checkpointLoaded
        ];
    } elseif (
        is_array($checkpointLoaded) &&
        empty($checkpointLoaded['finished'])
    ) {
        $state = $checkpointLoaded;
        $resumed = true;

        $batchSize = max(
            1,
            min(
                32,
                (int) (
                    $state['batch_size'] ??
                    $requestedBatchSize
                )
            )
        );
    } else {
        $batchSize = max(
            1,
            min(32, $requestedBatchSize)
        );

        $state = initialTrainingState(
            $sequences,
            $batchSize,
            $params,
            $cfg,
            $fineTune
        );

        if ($fineTune) {
            $state['fine_tune'] = true;
            $state['started_from_existing_weights'] = true;
        }
    }

    if (!is_array($state)) {
        fail('وضعیت آموزش قابل بازیابی نیست.');
    }

    if (!isset($state['order']) || !is_array($state['order'])) {
        $state['order'] = [];
    }

    if (
        !isset($state['epoch_losses']) ||
        !is_array($state['epoch_losses'])
    ) {
        $state['epoch_losses'] = [];
    }

    $batchSize = max(
        1,
        min(32, (int) ($state['batch_size'] ?? $requestedBatchSize))
    );

    $bestLoss = is_numeric($state['best_loss'] ?? null)
        ? (float) $state['best_loss']
        : INF;

    $bestEpoch = (int) ($state['best_epoch'] ?? 0);
    $staleEpochs = (int) ($state['stale_epochs'] ?? 0);
    $globalStep = (int) ($state['global_step'] ?? 0);
    $startEpoch = (int) ($state['epoch'] ?? 1);

    $progress = [];
    $started = microtime(true);

    while ($startEpoch <= MAX_EPOCHS) {
        if (empty($state['order'])) {
            $state['order'] = array_keys($sequences);
            shuffle($state['order']);

            $state['batch_offset'] = 0;
            $state['current_epoch_loss'] = 0.0;
            $state['current_epoch_batches'] = 0;
            $state['epoch'] = $startEpoch;
            $state['batch_size'] = $batchSize;

            saveCheckpoint(
                $state,
                $params,
                $m,
                $v,
                $bestParams,
                $cfg
            );
        }

        $batches = array_chunk(
            $state['order'],
            $batchSize
        );

        $totalBatches = count($batches);

        $beginBatch = max(
            0,
            (int) ($state['batch_offset'] ?? 0)
        );

        for (
            $batchIndex = $beginBatch;
            $batchIndex < $totalBatches;
            $batchIndex++
        ) {
            $batchRows = $batches[$batchIndex];
            $batchGrad = zeroLike($params);
            $batchLoss = 0.0;
            $validSamples = 0;

            foreach ($batchRows as $sampleIndex) {
                $ids = $sequences[$sampleIndex];

                if (
                    count($ids) >
                    (int) $cfg['max_position_embeddings']
                ) {
                    $ids = array_slice(
                        $ids,
                        -(int) $cfg['max_position_embeddings']
                    );
                }

                if (count($ids) < 2) {
                    continue;
                }

                $fwd = forwardTransformer(
                    $ids,
                    $params,
                    $cfg
                );

                $batchLoss += backwardTransformer(
                    $fwd,
                    $params,
                    $batchGrad,
                    $cfg,
                    MTP_LOSS_WEIGHT
                );

                $validSamples++;
            }

            if ($validSamples > 0) {
                scaleNested(
                    $batchGrad,
                    1.0 / $validSamples
                );

                $norm = sqrt(sumSquares($batchGrad));

                if ($norm > 1.0) {
                    scaleNested(
                        $batchGrad,
                        1.0 / $norm
                    );
                }

                $globalStep++;

                $learningRate =
                    BASE_LEARNING_RATE *
                    min(1.0, $globalStep / 5.0);

                adamUpdate(
                    $params,
                    $batchGrad,
                    $m,
                    $v,
                    $globalStep,
                    $learningRate
                );

                $state['current_epoch_loss'] =
                    (float) ($state['current_epoch_loss'] ?? 0.0) +
                    ($batchLoss / $validSamples);

                $state['current_epoch_batches'] =
                    (int) ($state['current_epoch_batches'] ?? 0) +
                    1;
            }

            $state['global_step'] = $globalStep;
            $state['batch_offset'] = $batchIndex + 1;
            $state['epoch'] = $startEpoch;
            $state['batch_size'] = $batchSize;

            $state['best_loss'] = is_finite($bestLoss)
                ? $bestLoss
                : null;

            $state['best_epoch'] = $bestEpoch;
            $state['stale_epochs'] = $staleEpochs;

            saveCheckpoint(
                $state,
                $params,
                $m,
                $v,
                $bestParams,
                $cfg
            );
        }

        $batchesDone = max(
            1,
            (int) ($state['current_epoch_batches'] ?? 0)
        );

        $epochLoss =
            (float) ($state['current_epoch_loss'] ?? 0.0) /
            $batchesDone;

        $epochRecord = [
            'epoch' => $startEpoch,
            'loss' => $epochLoss,
            'batches' => $totalBatches
        ];

        $state['epoch_losses'][] = $epochRecord;
        $progress[] = $epochRecord;

        if ($epochLoss < $bestLoss - 0.0001) {
            $bestLoss = $epochLoss;
            $bestEpoch = $startEpoch;
            $bestParams = $params;
            $staleEpochs = 0;
        } else {
            $staleEpochs++;
        }

        $startEpoch++;

        $state['epoch'] = $startEpoch;
        $state['order'] = [];
        $state['batch_offset'] = 0;
        $state['current_epoch_loss'] = 0.0;
        $state['current_epoch_batches'] = 0;
        $state['global_step'] = $globalStep;

        $state['best_loss'] = is_finite($bestLoss)
            ? $bestLoss
            : null;

        $state['best_epoch'] = $bestEpoch;
        $state['stale_epochs'] = $staleEpochs;

        saveCheckpoint(
            $state,
            $params,
            $m,
            $v,
            $bestParams,
            $cfg
        );

        $progress[] = [
            'epoch_complete' => $startEpoch - 1,
            'mean_loss' => $epochLoss,
            'best_loss' => $bestLoss,
            'elapsed_seconds' => microtime(true) - $started
        ];

        if ($staleEpochs >= EARLY_STOP_PATIENCE) {
            break;
        }
    }

    $params = $bestParams;

    $state['finished'] = true;
    $state['epoch'] = min(MAX_EPOCHS + 1, $startEpoch);
    $state['global_step'] = $globalStep;

    $state['best_loss'] = is_finite($bestLoss)
        ? $bestLoss
        : null;

    $state['best_epoch'] = $bestEpoch;
    $state['stale_epochs'] = $staleEpochs;

    saveCheckpoint(
        $state,
        $params,
        $m,
        $v,
        $bestParams,
        $cfg
    );

    $cfg['training_epochs'] = count($state['epoch_losses']);
    $cfg['best_epoch'] = $bestEpoch;

    $cfg['training_loss'] = is_finite($bestLoss)
        ? $bestLoss
        : null;

    $cfg['batch_size_last_training'] = $batchSize;
    $cfg['optimizer'] = 'Adam';
    $cfg['training_steps'] = $globalStep;
    $cfg['early_stopped'] =
        $staleEpochs >= EARLY_STOP_PATIENCE;

    $cfg['training_time_seconds'] =
        microtime(true) - $started;

    $cfg['epoch_losses'] = $state['epoch_losses'];
    $cfg['parameters'] = parameterCount($params);

    saveWeights($params, $cfg);
    saveConfig($cfg);

    @unlink(CHECKPOINT_FILE);

    return [
        'params' => $params,
        'config' => $cfg,
        'resumed' => $resumed,
        'progress' => $progress,
        'state' => $state
    ];
}

function samplingDistribution(
    array $logits,
    float $temperature,
    int $topK,
    float $topP,
    array $specialIds
): array {
    foreach ($specialIds as $id) {
        if (array_key_exists($id, $logits)) {
            $logits[$id] = -1.0e9;
        }
    }

    if ($temperature <= 0.0) {
        $bestId = 0;
        $bestValue = -INF;

        foreach ($logits as $id => $value) {
            if ($value > $bestValue) {
                $bestValue = $value;
                $bestId = (int) $id;
            }
        }

        return [$bestId => 1.0];
    }

    foreach ($logits as &$value) {
        $value = (float) $value / $temperature;
    }

    unset($value);

    $pairs = [];

    foreach ($logits as $id => $value) {
        $pairs[] = [
            'id' => (int) $id,
            'logit' => (float) $value
        ];
    }

    usort(
        $pairs,
        static fn(array $a, array $b): int =>
            $b['logit'] <=> $a['logit']
    );

    if ($topK > 0 && count($pairs) > $topK) {
        $pairs = array_slice($pairs, 0, $topK);
    }

    $probabilities = softmax(
        array_column($pairs, 'logit')
    );

    $kept = [];
    $cumulative = 0.0;

    foreach ($pairs as $i => $pair) {
        $kept[] = [
            'id' => $pair['id'],
            'p' => (float) $probabilities[$i]
        ];

        $cumulative += (float) $probabilities[$i];

        if (
            $topP > 0.0 &&
            $topP < 1.0 &&
            $cumulative >= $topP
        ) {
            break;
        }
    }

    $total = 0.0;

    foreach ($kept as $item) {
        $total += $item['p'];
    }

    if ($total <= 0.0) {
        return [(int) $pairs[0]['id'] => 1.0];
    }

    $distribution = [];

    foreach ($kept as $item) {
        $distribution[$item['id']] = $item['p'] / $total;
    }

    return $distribution;
}

function sampleFromDistribution(array $distribution): int
{
    $sum = array_sum($distribution);

    if ($sum <= 0.0) {
        return (int) array_key_first($distribution);
    }

    $random = (mt_rand() / mt_getrandmax()) * $sum;
    $cumulative = 0.0;
    $last = (int) array_key_first($distribution);

    foreach ($distribution as $id => $probability) {
        $last = (int) $id;
        $cumulative += (float) $probability;

        if ($random <= $cumulative) {
            return $last;
        }
    }

    return $last;
}

function sampleToken(
    array $logits,
    float $temperature,
    int $topK,
    float $topP,
    array $specialIds
): int {
    return sampleFromDistribution(
        samplingDistribution(
            $logits,
            $temperature,
            $topK,
            $topP,
            $specialIds
        )
    );
}

function rejectionVerify(
    array $draftDistribution,
    array $targetDistribution
): array {
    $candidate = sampleFromDistribution($draftDistribution);

    $q = (float) ($draftDistribution[$candidate] ?? 0.0);
    $p = (float) ($targetDistribution[$candidate] ?? 0.0);

    $acceptProbability = $q > 0.0
        ? min(1.0, $p / $q)
        : 0.0;

    if ((mt_rand() / mt_getrandmax()) < $acceptProbability) {
        return [
            'token' => $candidate,
            'accepted' => true
        ];
    }

    $residual = [];

    $ids = array_unique(
        array_merge(
            array_keys($targetDistribution),
            array_keys($draftDistribution)
        )
    );

    $sum = 0.0;

    foreach ($ids as $id) {
        $probability = max(
            0.0,
            (float) ($targetDistribution[$id] ?? 0.0) -
            (float) ($draftDistribution[$id] ?? 0.0)
        );

        if ($probability > 0.0) {
            $residual[(int) $id] = $probability;
            $sum += $probability;
        }
    }

    if ($sum <= 1.0e-12) {
        return [
            'token' => sampleFromDistribution($targetDistribution),
            'accepted' => false
        ];
    }

    return [
        'token' => sampleFromDistribution($residual),
        'accepted' => false
    ];
}

function generateAnswer(
    string $question,
    array $tokenizer,
    array $params,
    array $cfg,
    bool $mtpEnabled,
    float $temperature,
    int $topK,
    float $topP
): array {
    $prompt = 'User: ' . $question . "\nAssistant:";

    $tokenStart = microtime(true);

    $questionIds = bpeEncode($question, $tokenizer);
    $unknownId = (int) $tokenizer['vocab'][UNK];
    $unknownInputTokens = 0;

    foreach ($questionIds as $questionId) {
        if ((int) $questionId === $unknownId) {
            $unknownInputTokens++;
        }
    }

    $promptIds = array_merge(
        [(int) $tokenizer['vocab'][BOS]],
        bpeEncode($prompt, $tokenizer)
    );

    $tokenizeSeconds = max(
        1.0e-9,
        microtime(true) - $tokenStart
    );

    $history = $promptIds;
    $generated = [];

    $accepted = 0;
    $rejected = 0;
    $draftCount = 0;
    $prefillTokens = 0;

    $bosId = (int) $tokenizer['vocab'][BOS];
    $padId = (int) $tokenizer['vocab'][PAD];
    $unkId = (int) $tokenizer['vocab'][UNK];
    $eosId = (int) $tokenizer['vocab'][EOS];

    $maskIds = [$bosId, $padId, $unkId];

    $generationLimit = (int) $cfg['generation_limit'];
    $generationStart = microtime(true);

    while (count($generated) < $generationLimit) {
        $stepMaskIds = $maskIds;

        if (count($generated) < min(2, $generationLimit)) {
            $stepMaskIds[] = $eosId;
        }

        $contextLimit = (int) $cfg['max_position_embeddings'];

        $window = count($history) > $contextLimit
            ? array_slice($history, -$contextLimit)
            : $history;

        $fwd = forwardTransformer(
            $window,
            $params,
            $cfg
        );

        $prefillTokens += count($window);
        $last = count($fwd['logits']) - 1;

        $mainDistribution = samplingDistribution(
            $fwd['logits'][$last],
            $temperature,
            $topK,
            $topP,
            $stepMaskIds
        );

        $next = sampleFromDistribution($mainDistribution);

        if ($next === $eosId) {
            break;
        }

        $history[] = $next;
        $generated[] = $next;

        if (
            !$mtpEnabled ||
            count($generated) >= $generationLimit
        ) {
            continue;
        }

        $draftCount++;

        $draftDistribution = samplingDistribution(
            $fwd['mtp_logits'][$last],
            $temperature,
            $topK,
            $topP,
            $stepMaskIds
        );

        $verifyWindow = count($history) > $contextLimit
            ? array_slice($history, -$contextLimit)
            : $history;

        $verifyFwd = forwardTransformer(
            $verifyWindow,
            $params,
            $cfg
        );

        $prefillTokens += count($verifyWindow);

        $verifyLast = count($verifyFwd['logits']) - 1;

        $targetDistribution = samplingDistribution(
            $verifyFwd['logits'][$verifyLast],
            $temperature,
            $topK,
            $topP,
            $stepMaskIds
        );

        $decision = rejectionVerify(
            $draftDistribution,
            $targetDistribution
        );

        $nextTwo = (int) $decision['token'];

        if ($decision['accepted']) {
            $accepted++;
        } else {
            $rejected++;
        }

        if ($nextTwo === $eosId) {
            break;
        }

        $history[] = $nextTwo;
        $generated[] = $nextTwo;
    }

    $generationSeconds = max(
        1.0e-9,
        microtime(true) - $generationStart
    );

    return [
        'answer' => decodeIds($generated, $tokenizer),
        'prompt_ids' => $promptIds,
        'output_ids' => $generated,
        'tokenize_seconds' => $tokenizeSeconds,
        'tokenize_rate' => count($promptIds) / $tokenizeSeconds,
        'generation_seconds' => $generationSeconds,
        'generation_rate' => count($generated) / $generationSeconds,
        'prefill_rate' => $prefillTokens / $generationSeconds,
        'mtp_accepted' => $accepted,
        'mtp_rejected' => $rejected,
        'mtp_drafts' => $draftCount,
        'input_tokens' => count($promptIds),
        'question_tokens' => count($questionIds),
        'unknown_input_tokens' => $unknownInputTokens,
        'context_tokens' => min(
            count($promptIds),
            (int) $cfg['max_position_embeddings']
        ),
        'prefill_tokens' => $prefillTokens,
        'output_tokens' => count($generated)
    ];
}

function visibleTokens(array $ids, array $tokenizer): string
{
    $tokens = [];

    foreach ($ids as $id) {
        $tokens[] = json_encode(
            tokenString((int) $id, $tokenizer),
            JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES
        );
    }

    return implode(' ', $tokens);
}

function safeInt(
    string $key,
    int $default,
    int $min,
    int $max
): int {
    if (!isset($_GET[$key]) || !is_scalar($_GET[$key])) {
        return $default;
    }

    $value = filter_var(
        (string) $_GET[$key],
        FILTER_VALIDATE_INT
    );

    if ($value === false) {
        return $default;
    }

    return max($min, min($max, (int) $value));
}

function safeFloat(
    string $key,
    float $default,
    float $min,
    float $max
): float {
    if (!isset($_GET[$key]) || !is_scalar($_GET[$key])) {
        return $default;
    }

    $value = filter_var(
        (string) $_GET[$key],
        FILTER_VALIDATE_FLOAT
    );

    if ($value === false) {
        return $default;
    }

    return max($min, min($max, (float) $value));
}

function removeTrainingArtifacts(): void
{
    foreach (
        [
            WEIGHTS_FILE,
            CHECKPOINT_FILE,
            TRAINING_STATE_FILE,
            VOCAB_FILE,
            MERGES_FILE,
            TOKENIZER_CONFIG_FILE,
            TOKENIZER_FILE,
            CONFIG_FILE,
            MODEL_CONFIG_FILE
        ] as $file
    ) {
        @unlink($file);
    }
}

function fileIdentityMatches(
    string $path,
    string $magic,
    array $cfg,
    int $hashOffset = 8
): bool {
    if (!is_file($path)) {
        return false;
    }

    $fp = fopen($path, 'rb');

    if ($fp === false) {
        return false;
    }

    $actualMagic = fread($fp, strlen($magic));
    $hash = fread($fp, 32);

    fclose($fp);

    return
        $actualMagic === $magic &&
        strlen($hash) === 32 &&
        hash_equals(
            hex2bin(identityHash($cfg)),
            $hash
        );
}

function exactArchitecture(array $cfg): bool
{
    return
        ($cfg['architectures'] ?? null) ===
            ['JibayNanoForCausalLM'] &&
        ($cfg['model_type'] ?? null) === 'jibay_nano';
}

function reportModel(array $cfg): void
{
    echo "Jibay Nano causal language model\n";

    echo 'Model type: ' .
        $cfg['model_type'] . "\n";

    echo 'Architectures: ' .
        implode(', ', $cfg['architectures']) . "\n";

    echo 'Layers: ' . $cfg['n_layers'] .
        ', attention heads: ' . $cfg['n_heads'] .
        ', d_model: ' . $cfg['d_model'] .
        ', FFN: ' . $cfg['ff_dim'] . "\n";

    echo 'Vocab size: ' . $cfg['vocab_size'] .
        ', context length: ' . $cfg['max_position_embeddings'] .
        ', generation limit: ' . $cfg['generation_limit'] . "\n";

    echo 'Trainable parameters: ' .
        number_format((int) $cfg['parameters']) . "\n";

    echo 'Training epochs: ' .
        (int) $cfg['training_epochs'] . '/' . MAX_EPOCHS .
        ', best epoch: ' . (int) $cfg['best_epoch'] .
        ', loss: ' . (string) ($cfg['training_loss'] ?? 'unknown') .
        "\n";
}

try {
    $rows = readDataset(DATASET_FILE);

    $datasetHash = hash_file(
        'sha256',
        DATASET_FILE
    );

    if (!is_string($datasetHash)) {
        fail('محاسبهٔ fingerprint دیتاست ناموفق بود.');
    }

    $question = isset($_GET['t']) && is_scalar($_GET['t'])
        ? trim((string) $_GET['t'])
        : '';

    if (strlen($question) > 12000) {
        fail('ورودی از حد مجاز بزرگ‌تر است.', 413);
    }

    $requestedBatch = safeInt(
        'batch',
        8,
        1,
        32
    );

    $temperature = safeFloat(
        'temperature',
        0.8,
        0.0,
        3.0
    );

    $topK = safeInt(
        'top_k',
        40,
        0,
        1000
    );

    $topP = safeFloat(
        'top_p',
        0.9,
        0.0,
        1.0
    );

    $mtpEnabled =
        !isset($_GET['mtp']) ||
        (string) $_GET['mtp'] !== '0';

    $forceTrain =
        isset($_GET['train']) &&
        (string) $_GET['train'] === '1';

    $lock = fopen(LOCK_FILE, 'c');

    if (
        $lock === false ||
        !flock($lock, LOCK_EX)
    ) {
        fail('قفل آموزش باز نشد؛ مجوز نوشتن پوشه را بررسی کن.');
    }

    $cfg = readConfig();

    $hasWeights = is_file(WEIGHTS_FILE);
    $hasCheckpoint = is_file(CHECKPOINT_FILE);
    $hasArtifacts = $hasWeights || $hasCheckpoint;

    if ($cfg !== null && !exactArchitecture($cfg)) {
        if ($hasArtifacts) {
            fail(
                'معماری به وزن‌ها متصل است. architectures و model_type را تغییر نده؛ برای تغییر معماری باید عمداً وزن‌ها و checkpoint را حذف و مدل را از نو بسازی.'
            );
        }

        $cfg = null;
    }

    if (
        $cfg !== null &&
        ($cfg['dataset_sha256'] ?? '') !== $datasetHash
    ) {
        removeTrainingArtifacts();

        $cfg = null;
        $hasWeights = false;
        $hasCheckpoint = false;
    }

    $tokenizer = null;

    if ($cfg !== null) {
        $tokenizerFilesExist =
            is_file(VOCAB_FILE) &&
            is_file(MERGES_FILE) &&
            is_file(TOKENIZER_CONFIG_FILE) &&
            is_file(TOKENIZER_FILE);

        if (!$tokenizerFilesExist) {
            if ($hasArtifacts) {
                fail(
                    'فایل توکنیزر با وزن‌های موجود ناقص است؛ فایل‌ها را بازیابی کن یا همهٔ مدل را عمداً از نو بساز.'
                );
            }

            $cfg = null;
        } else {
            $tokenizer = loadTokenizer();

            $actualTokenizerHash = tokenizerFingerprint();

            if (
                ($cfg['tokenizer_sha256'] ?? '') !==
                $actualTokenizerHash
            ) {
                if ($hasArtifacts) {
                    fail(
                        'توکنیزر با وزن‌های موجود مطابقت ندارد. فایل‌های vocab و merges را دستی تغییر نده.'
                    );
                }

                $cfg = null;
            }
        }
    }

    if ($cfg === null) {
        if (
            is_file(WEIGHTS_FILE) ||
            is_file(CHECKPOINT_FILE)
        ) {
            fail(
                'config.json وجود ندارد یا ناسازگار است، اما وزن یا checkpoint موجود است. config را از نسخهٔ پشتیبان برگردان.'
            );
        }

        foreach (
            [
                VOCAB_FILE,
                MERGES_FILE,
                TOKENIZER_CONFIG_FILE,
                TOKENIZER_FILE
            ] as $file
        ) {
            @unlink($file);
        }

        $texts = [];

        foreach ($rows as $row) {
            $texts[] = trainingText($row);
        }

        $trainedBpe = trainBpe($texts);

        saveTokenizer($trainedBpe);

        $tokenizer = loadTokenizer();

        $tokenizerHash = tokenizerFingerprint();

        $cfg = deriveModelConfig(
            $rows,
            $tokenizer,
            $datasetHash,
            $tokenizerHash
        );

        $temporaryParams = initParams($cfg);

        $cfg['parameters'] = parameterCount(
            $temporaryParams
        );

        unset($temporaryParams);

        saveConfig($cfg);
    }

    if (!is_array($tokenizer)) {
        $tokenizer = loadTokenizer();
    }

    if (
        ($cfg['dataset_sha256'] ?? '') !== $datasetHash ||
        ($cfg['tokenizer_sha256'] ?? '') !== tokenizerFingerprint()
    ) {
        fail('دیتاست یا توکنیزر بعد از پیکربندی تغییر کرده است.');
    }

    if (
        $hasWeights &&
        !fileIdentityMatches(
            WEIGHTS_FILE,
            MODEL_MAGIC,
            $cfg
        )
    ) {
        if (!$hasCheckpoint) {
            fail(
                'هش معماری config.json با هدر model_weights.bin یکسان نیست؛ مدل و تنظیمات با هم سازگار نیستند.'
            );
        }
    }

    if (
        $hasCheckpoint &&
        !fileIdentityMatches(
            CHECKPOINT_FILE,
            CHECKPOINT_MAGIC,
            $cfg
        )
    ) {
        if ($hasWeights && !$forceTrain) {
            fail('checkpoint با معماری یا دیتاست فعلی سازگار نیست.');
        }

        @unlink(CHECKPOINT_FILE);
        @unlink(TRAINING_STATE_FILE);

        $hasCheckpoint = false;
    }

    $needsTraining =
        $forceTrain ||
        !$hasWeights ||
        is_file(CHECKPOINT_FILE);

    $trainedNow = false;
    $resumed = false;
    $trainingProgress = [];

    if ($needsTraining) {
        $params = initParams($cfg);

        if ($hasWeights) {
            if (!loadWeights($params, $cfg)) {
                fail('بارگذاری وزن‌ها ناموفق بود.');
            }
        }

        $sequencesInfo = buildTrainingSequences(
            $rows,
            $tokenizer
        );

        $result = trainModel(
            $sequencesInfo['sequences'],
            $params,
            $cfg,
            $requestedBatch,
            $forceTrain
        );

        $params = $result['params'];
        $cfg = $result['config'];

        $resumed = $result['resumed'];
        $trainingProgress = $result['progress'];
        $trainedNow = true;
    } else {
        $params = initParams($cfg);

        if (!loadWeights($params, $cfg)) {
            fail(
                'وزن‌ها وجود دارند اما بارگذاری نشدند؛ معماری فایل config.json را با وزن‌ها تغییر نده.'
            );
        }
    }

    flock($lock, LOCK_UN);
    fclose($lock);

    if ($question === '') {
        reportModel($cfg);

        echo 'Dataset examples: ' .
            count($rows) . "\n";

        echo 'Loaded weights from disk: ' .
            ($trainedNow
                ? 'training finished in this request'
                : 'yes') . "\n";

        echo 'Training resumed from checkpoint: ' .
            ($resumed ? 'yes' : 'no') . "\n";

        echo 'Batch size: ' .
            (string) (
                $cfg['batch_size_last_training'] ??
                $requestedBatch
            ) . "\n";

        echo 'Temporary checkpoint: ' .
            (is_file(CHECKPOINT_FILE)
                ? 'present'
                : 'not present') . "\n";

        echo 'Files: config.json, model_config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, model_weights.bin, training_state.json' . "\n";

        echo "Use ?t=سلام to generate. Add &mtp=0 to disable MTP, &batch=8 to set batch size, or &train=1 to continue training from saved weights.\n";

        if ($trainingProgress) {
            echo "Training progress in this request:\n";

            foreach ($trainingProgress as $entry) {
                if (isset($entry['epoch'])) {
                    echo 'Epoch ' .
                        $entry['epoch'] .
                        ' loss=' .
                        number_format(
                            (float) $entry['loss'],
                            5,
                            '.',
                            ''
                        ) . "\n";
                } elseif (isset($entry['epoch_complete'])) {
                    echo 'Completed epoch ' .
                        $entry['epoch_complete'] .
                        ' mean_loss=' .
                        number_format(
                            (float) $entry['mean_loss'],
                            5,
                            '.',
                            ''
                        ) . "\n";
                }
            }
        }

        exit;
    }

    $answerResult = generateAnswer(
        $question,
        $tokenizer,
        $params,
        $cfg,
        $mtpEnabled,
        $temperature,
        $topK,
        $topP
    );

    echo "Question: {$question}\n";
    echo "Answer:\n" . $answerResult['answer'] . "\n\n";

    echo 'Question tokens: ' .
        $answerResult['question_tokens'] . "\n";

    echo 'Prompt tokens including control tokens: ' .
        $answerResult['input_tokens'] . "\n";

    echo 'Active context tokens: ' .
        $answerResult['context_tokens'] .
        ' / ' .
        $cfg['max_position_embeddings'] . "\n";

    echo 'Unknown input tokens: ' .
        $answerResult['unknown_input_tokens'] . "\n";

    echo 'Verified prefill tokens: ' .
        $answerResult['prefill_tokens'] . "\n";

    echo 'Input tokens (' .
        $answerResult['input_tokens'] .
        '):' . "\n" .
        visibleTokens(
            $answerResult['prompt_ids'],
            $tokenizer
        ) . "\n\n";

    echo 'Output tokens (' .
        $answerResult['output_tokens'] .
        '):' . "\n" .
        visibleTokens(
            $answerResult['output_ids'],
            $tokenizer
        ) . "\n\n";

    echo 'Tokenization speed: ' .
        number_format(
            $answerResult['tokenize_rate'],
            2,
            '.',
            ''
        ) . " tokens/sec\n";

    echo 'Context/prefill rate: ' .
        number_format(
            $answerResult['prefill_rate'],
            2,
            '.',
            ''
        ) . " tokens/sec\n";

    echo 'Generation speed: ' .
        number_format(
            $answerResult['generation_rate'],
            2,
            '.',
            ''
        ) . " tokens/sec\n";

    echo 'Parameters: ' .
        number_format((int) $cfg['parameters']) . "\n";

    echo 'Architecture: ' .
        implode(', ', $cfg['architectures']) . "\n";

    echo 'Model type: ' .
        $cfg['model_type'] . "\n";

    echo 'Layers: ' .
        $cfg['n_layers'] .
        ', heads: ' .
        $cfg['n_heads'] .
        ', d_model: ' .
        $cfg['d_model'] .
        ', FFN: ' .
        $cfg['ff_dim'] . "\n";

    $tokenizerConfig = json_decode(
        (string) file_get_contents(TOKENIZER_CONFIG_FILE),
        true
    );

    echo 'Vocabulary: ' .
        $cfg['vocab_size'] .
        ', BPE merges: ' .
        (int) ($tokenizerConfig['merge_count'] ?? 0) .
        "\n";

    echo 'Auto context: ' .
        $cfg['max_position_embeddings'] .
        ', auto generation limit: ' .
        $cfg['generation_limit'] . "\n";

    echo 'Sampling: temperature=' .
        $temperature .
        ', top_k=' .
        $topK .
        ', top_p=' .
        $topP . "\n";

    echo 'MTP: ' .
        ($mtpEnabled ? 'enabled' : 'disabled') .
        ', proposed=' .
        $answerResult['mtp_drafts'] .
        ', accepted=' .
        $answerResult['mtp_accepted'] .
        ', rejected=' .
        $answerResult['mtp_rejected'] .
        "\n";

    echo 'Training: epochs=' .
        $cfg['training_epochs'] .
        '/' .
        MAX_EPOCHS .
        ', best_epoch=' .
        $cfg['best_epoch'] .
        ', loss=' .
        (string) ($cfg['training_loss'] ?? 'unknown') .
        ', batch_size=' .
        (string) (
            $cfg['batch_size_last_training'] ??
            $requestedBatch
        ) .
        "\n";

    echo 'Model trained in this request: ' .
        ($trainedNow ? 'yes' : 'no') . "\n";

    echo 'Training resumed from checkpoint: ' .
        ($resumed ? 'yes' : 'no') . "\n";

} catch (Throwable $e) {
    fail('خطای اجرایی: ' . $e->getMessage());
}