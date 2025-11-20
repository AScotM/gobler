package main

import (
	crand "crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"os"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

type MarkovSeedGenerator struct {
	N             int
	Model         map[string][]rune
	Verbose       bool
	UseCryptoRand bool
	logMessages   []string
	mu            sync.RWMutex
	logMu         sync.Mutex
}

type ModelStats struct {
	NGrams           int
	TotalTransitions int
	AvgTransitions   float64
	MaxTransitions   int
	MinTransitions   int
	DeadEnds         int
}

func NewMarkovSeedGenerator(n int, verbose bool) (*MarkovSeedGenerator, error) {
	if n <= 0 {
		return nil, fmt.Errorf("n must be positive")
	}
	return &MarkovSeedGenerator{
		N:            n,
		Model:        make(map[string][]rune),
		Verbose:      verbose,
		UseCryptoRand: true,
		logMessages:  make([]string, 0),
	}, nil
}

func secureRandIntn(n int) int {
	if n <= 0 {
		return 0
	}
	num, err := crand.Int(crand.Reader, big.NewInt(int64(n)))
	if err != nil {
		var fallback int64
		binary.Read(crand.Reader, binary.BigEndian, &fallback)
		if fallback < 0 {
			fallback = -fallback
		}
		return int(fallback % int64(n))
	}
	return int(num.Int64())
}

func (m *MarkovSeedGenerator) randIntn(n int) int {
	if m.UseCryptoRand {
		return secureRandIntn(n)
	}
	return secureRandIntn(n)
}

func (m *MarkovSeedGenerator) log(format string, args ...interface{}) {
	if !m.Verbose {
		return
	}
	message := fmt.Sprintf(format, args...)
	m.logMu.Lock()
	m.logMessages = append(m.logMessages, message)
	m.logMu.Unlock()
	log.Println(message)
}

func (m *MarkovSeedGenerator) GetLogs() []string {
	m.logMu.Lock()
	defer m.logMu.Unlock()
	logs := make([]string, len(m.logMessages))
	copy(logs, m.logMessages)
	return logs
}

func (m *MarkovSeedGenerator) ClearLogs() {
	m.logMu.Lock()
	defer m.logMu.Unlock()
	m.logMessages = m.logMessages[:0]
}

func sanitizeText(text string) string {
	return strings.Map(func(r rune) rune {
		if r == '\t' || r == '\n' || r == '\r' {
			return r
		}
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, text)
}

func (m *MarkovSeedGenerator) Train(text string) error {
	text = sanitizeText(text)
	runes := []rune(text)
	if len(runes) <= m.N {
		return fmt.Errorf("text length %d must be greater than n %d", len(runes), m.N)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for i := 0; i <= len(runes)-m.N; i++ {
		key := string(runes[i : i+m.N])
		if i+m.N < len(runes) {
			nextChar := runes[i+m.N]
			m.Model[key] = append(m.Model[key], nextChar)
		}
	}

	m.log("Trained model with %d n-grams", len(m.Model))
	return nil
}

func (m *MarkovSeedGenerator) TrainFromFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open training file: %w", err)
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file info: %w", err)
	}

	if info.Size() == 0 {
		return fmt.Errorf("training file is empty")
	}

	if info.Size() > 100*1024*1024 {
		return fmt.Errorf("file too large: %d bytes", info.Size())
	}

	m.log("Training from file: %s", filename)

	var textBuilder strings.Builder
	_, err = io.Copy(&textBuilder, file)
	if err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	return m.Train(textBuilder.String())
}

func (m *MarkovSeedGenerator) Generate(length int, startWith ...string) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.Model) == 0 {
		return "", fmt.Errorf("untrained model")
	}
	if length < m.N {
		return "", fmt.Errorf("length %d must be at least n %d", length, m.N)
	}

	keys := make([]string, 0, len(m.Model))
	for k := range m.Model {
		keys = append(keys, k)
	}

	var seed string
	if len(startWith) > 0 {
		input := startWith[0]
		if utf8.RuneCountInString(input) >= m.N {
			inputRunes := []rune(input)
			seed = string(inputRunes[:m.N])
		}
	}

	if seed == "" || len(m.Model[seed]) == 0 {
		seed = keys[m.randIntn(len(keys))]
		if len(startWith) > 0 {
			m.log("Warning: Starting text %q not found, using random n-gram", startWith[0])
		}
	} else {
		m.log("Starting generation with: %q", seed)
	}

	output := []rune(seed)

	for len(output) < length {
		nextChars := m.Model[seed]
		if len(nextChars) == 0 {
			similar := m.findSimilarNgram(seed)
			if similar != "" && len(m.Model[similar]) > 0 {
				m.log("Fallback: using similar n-gram %q for %q", similar, seed)
				nextChars = m.Model[similar]
			} else {
				nextChars = m.Model[keys[m.randIntn(len(keys))]]
			}
		}

		if len(nextChars) == 0 {
			return "", fmt.Errorf("no valid transitions available")
		}

		nextChar := nextChars[m.randIntn(len(nextChars))]
		output = append(output, nextChar)

		seedRunes := []rune(seed)
		seed = string(seedRunes[1:]) + string(nextChar)
	}

	return string(output[:length]), nil
}

func (m *MarkovSeedGenerator) findSimilarNgram(target string) string {
	targetRunes := []rune(target)
	bestMatch := ""
	bestDistance := int(^uint(0) >> 1)

	for key := range m.Model {
		if len(m.Model[key]) == 0 {
			continue
		}
		keyRunes := []rune(key)
		if len(keyRunes) != len(targetRunes) {
			continue
		}
		distance := simpleRuneDistance(targetRunes, keyRunes)
		if distance < bestDistance {
			bestDistance = distance
			bestMatch = key
		}
		if bestDistance == 0 {
			break
		}
	}

	return bestMatch
}

func simpleRuneDistance(a, b []rune) int {
	if len(a) != len(b) {
		return int(^uint(0) >> 1)
	}
	distance := 0
	for i := range a {
		if a[i] != b[i] {
			distance++
		}
	}
	return distance
}

func (m *MarkovSeedGenerator) ValidateModel() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.N <= 0 {
		return fmt.Errorf("invalid n value: %d", m.N)
	}

	for key, transitions := range m.Model {
		if utf8.RuneCountInString(key) != m.N {
			return fmt.Errorf("invalid key length: %q (expected %d)", key, m.N)
		}
		if len(transitions) == 0 {
			return fmt.Errorf("key %q has no transitions", key)
		}
	}
	return nil
}

func (m *MarkovSeedGenerator) GetModelStats() ModelStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := ModelStats{
		MinTransitions: -1,
	}

	for _, transitions := range m.Model {
		count := len(transitions)
		stats.NGrams++
		stats.TotalTransitions += count

		if count > stats.MaxTransitions {
			stats.MaxTransitions = count
		}
		if stats.MinTransitions == -1 || count < stats.MinTransitions {
			stats.MinTransitions = count
		}
		if count == 0 {
			stats.DeadEnds++
		}
	}

	if stats.NGrams > 0 {
		stats.AvgTransitions = float64(stats.TotalTransitions) / float64(stats.NGrams)
	}

	return stats
}

func (m *MarkovSeedGenerator) SaveModel(filename string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(m.Model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	m.log("Model saved to %s", filename)
	return nil
}

func (m *MarkovSeedGenerator) LoadModel(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	var model map[string][]rune
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	for key := range model {
		if utf8.RuneCountInString(key) != m.N {
			return fmt.Errorf("loaded model key %q has length != %d", key, m.N)
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.Model = model

	m.log("Model loaded from %s with %d n-grams", filename, len(m.Model))
	return nil
}

func (m *MarkovSeedGenerator) GetAvailableKeys() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	keys := make([]string, 0, len(m.Model))
	for k := range m.Model {
		keys = append(keys, k)
	}
	return keys
}

func (m *MarkovSeedGenerator) GetTransitions(key string) []rune {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.Model[key]
}

func (m *MarkovSeedGenerator) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Model = make(map[string][]rune)
	m.ClearLogs()
}

func main() {
	const trainingText = `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>/?` +
		`The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.`

	markov, err := NewMarkovSeedGenerator(3, true)
	if err != nil {
		log.Fatal(err)
	}

	if err := markov.Train(trainingText); err != nil {
		log.Fatal(err)
	}

	if err := markov.ValidateModel(); err != nil {
		log.Printf("Model validation warning: %v", err)
	}

	stats := markov.GetModelStats()
	fmt.Printf("Model Statistics:\n")
	fmt.Printf("- N-Grams: %d\n", stats.NGrams)
	fmt.Printf("- Total Transitions: %d\n", stats.TotalTransitions)
	fmt.Printf("- Average Transitions: %.2f\n", stats.AvgTransitions)
	fmt.Printf("- Max Transitions: %d\n", stats.MaxTransitions)
	fmt.Printf("- Min Transitions: %d\n", stats.MinTransitions)
	fmt.Printf("- Dead Ends: %d\n", stats.DeadEnds)
	fmt.Println()

	for i := 0; i < 5; i++ {
		seed, err := markov.Generate(16)
		if err != nil {
			log.Println("Error:", err)
			continue
		}
		fmt.Printf("Generated %d: %q\n", i+1, seed)
	}

	fmt.Println()

	seeded, err := markov.Generate(20, "The")
	if err != nil {
		log.Println("Error:", err)
	} else {
		fmt.Printf("Seeded generation: %q\n", seeded)
	}

	if err := markov.SaveModel("markov_model.json"); err != nil {
		log.Println("Error saving model:", err)
	}

	markov2, err := NewMarkovSeedGenerator(3, true)
	if err != nil {
		log.Fatal(err)
	}
	if err := markov2.LoadModel("markov_model.json"); err != nil {
		log.Fatal(err)
	}
	reloaded, err := markov2.Generate(16)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("From reloaded model: %q\n", reloaded)
}
