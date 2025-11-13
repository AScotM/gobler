package main

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"os"
	"strings"
	"unicode/utf8"
)

type MarkovSeedGenerator struct {
	N           int
	Model       map[string][]rune
	Text        string
	Verbose     bool
	logMessages []string
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
		N:           n,
		Model:       make(map[string][]rune),
		Verbose:     verbose,
		logMessages: make([]string, 0),
	}, nil
}

func secureRandIntn(n int) int {
	if n <= 0 {
		return 0
	}
	num, err := rand.Int(rand.Reader, big.NewInt(int64(n)))
	if err != nil {
		var fallback int64
		if err := binary.Read(rand.Reader, binary.BigEndian, &fallback); err != nil {
			return int(big.NewInt(0).Mod(big.NewInt(int64(os.Getpid())^int64(n)), big.NewInt(int64(n))).Int64())
		}
		if fallback < 0 {
			fallback = -fallback
		}
		return int(fallback % int64(n))
	}
	return int(num.Int64())
}

func (m *MarkovSeedGenerator) log(format string, args ...interface{}) {
	message := fmt.Sprintf(format, args...)
	if m.Verbose {
		m.logMessages = append(m.logMessages, message)
		log.Println(message)
	}
}

func (m *MarkovSeedGenerator) GetLogs() []string {
	return m.logMessages
}

func (m *MarkovSeedGenerator) ClearLogs() {
	m.logMessages = m.logMessages[:0]
}

func (m *MarkovSeedGenerator) Train(text string) error {
	text = sanitizeText(text)
	
	runes := []rune(text)
	if len(runes) <= m.N {
		return fmt.Errorf("text length %d must be greater than n %d", len(runes), m.N)
	}

	m.Text = text
	limit := len(runes) - m.N

	for i := 0; i < limit; i++ {
		end := i + m.N
		if end >= len(runes) {
			break
		}
		key := string(runes[i:end])
		nextChar := runes[end]
		
		m.Model[key] = append(m.Model[key], nextChar)
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

	m.log("Training from file: %s", filename)

	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file info: %w", err)
	}
	fileSize := info.Size()

	buffer := make([]byte, 8192)
	var textBuilder strings.Builder
	processedBytes := int64(0)

	for {
		n, err := file.Read(buffer)
		if err != nil && err != io.EOF {
			return fmt.Errorf("error reading file: %w", err)
		}

		if n == 0 {
			break
		}

		chunk := sanitizeText(string(buffer[:n]))
		textBuilder.WriteString(chunk)
		processedBytes += int64(n)

		if m.Verbose && fileSize > 0 {
			percent := float64(processedBytes) / float64(fileSize) * 100
			m.log("Processed %d/%d bytes (%.1f%%)", processedBytes, fileSize, percent)
		}

		if err == io.EOF {
			break
		}
	}

	return m.Train(textBuilder.String())
}

func sanitizeText(text string) string {
	var result strings.Builder
	for _, r := range text {
		if r >= 32 && r != 127 || r == '\n' || r == '\t' {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func (m *MarkovSeedGenerator) Generate(length int, startWith ...string) (string, error) {
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
	if len(startWith) > 0 && utf8.RuneCountInString(startWith[0]) == m.N {
		if _, exists := m.Model[startWith[0]]; exists {
			seed = startWith[0]
			m.log("Starting generation with: %q", seed)
		} else {
			m.log("Warning: Starting n-gram %q not found, using random", startWith[0])
			seed = keys[secureRandIntn(len(keys))]
		}
	} else {
		seed = keys[secureRandIntn(len(keys))]
	}

	output := []rune(seed)

	for len(output) < length {
		nextChars := m.Model[seed]
		if len(nextChars) == 0 {
			similar := m.findSimilarNgram(seed)
			if similar != "" {
				m.log("Fallback: using similar n-gram %q for %q", similar, seed)
				nextChars = m.Model[similar]
			} else {
				runes := []rune(m.Text)
				if len(runes) == 0 {
					return "", fmt.Errorf("no text available for fallback")
				}
				nextChar := runes[secureRandIntn(len(runes))]
				output = append(output, nextChar)
				if len(seed) > 0 {
					seed = string([]rune(seed)[1:]) + string(nextChar)
				}
				continue
			}
		}

		if len(nextChars) == 0 {
			return "", fmt.Errorf("no valid transitions available")
		}

		nextChar := nextChars[secureRandIntn(len(nextChars))]
		output = append(output, nextChar)
		
		seedRunes := []rune(seed)
		seed = string(seedRunes[1:]) + string(nextChar)
	}

	return string(output[:length]), nil
}

func (m *MarkovSeedGenerator) findSimilarNgram(target string) string {
	bestMatch := ""
	bestDistance := -1
	targetRunes := []rune(target)

	for key := range m.Model {
		keyRunes := []rune(key)
		distance := levenshteinDistance(targetRunes, keyRunes)
		if bestDistance == -1 || distance < bestDistance {
			bestDistance = distance
			bestMatch = key
		}
		if bestDistance <= 1 {
			break
		}
	}

	return bestMatch
}

func levenshteinDistance(a, b []rune) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}

	matrix := make([][]int, len(a)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(b)+1)
		matrix[i][0] = i
	}
	for j := range matrix[0] {
		matrix[0][j] = j
	}

	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			matrix[i][j] = min(
				matrix[i-1][j]+1,
				matrix[i][j-1]+1,
				matrix[i-1][j-1]+cost,
			)
		}
	}

	return matrix[len(a)][len(b)]
}

func min(values ...int) int {
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func (m *MarkovSeedGenerator) ValidateModel() ModelStats {
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

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&m.Model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	m.log("Model loaded from %s with %d n-grams", filename, len(m.Model))
	return nil
}

func (m *MarkovSeedGenerator) GetAvailableKeys() []string {
	keys := make([]string, 0, len(m.Model))
	for k := range m.Model {
		keys = append(keys, k)
	}
	return keys
}

func (m *MarkovSeedGenerator) GetTransitions(key string) []rune {
	return m.Model[key]
}

func (m *MarkovSeedGenerator) Reset() {
	m.Model = make(map[string][]rune)
	m.Text = ""
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

	stats := markov.ValidateModel()
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
