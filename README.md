# GEPA-TS

TypeScript implementation of GEPA (Gradient-free Evolution of Prompts and Agents) with complete 1:1 feature parity to the Python version.

GEPA optimizes text components (prompts, instructions) using evolutionary algorithms and LLM-based reflection to automatically improve performance on your tasks.

## Installation

```bash
npm install gepa-ts
```

## Quick Start

```typescript
import { optimize } from 'gepa-ts';

// Define your training data
const trainData = [
  { input: "I love this product!", answer: "positive" },
  { input: "This is terrible", answer: "negative" },
  { input: "It's okay I guess", answer: "neutral" }
];

// Run optimization
const result = await optimize({
  seedCandidate: { 
    system_prompt: "Classify the sentiment of this text" 
  },
  trainset: trainData,
  maxMetricCalls: 50,
  reflectionLM: async (prompt) => {
    // Your LLM call for generating improvements
    return await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }]
    }).choices[0].message.content;
  }
});

console.log('Best prompt:', result.bestCandidate.system_prompt);
console.log('Score improvement:', result.bestScore);
```

## Key Features

- **Complete Python Parity**: Identical API, behavior, and results to Python GEPA
- **Per-Instance Pareto Optimization**: Tracks best candidates for each validation example
- **LLM-Based Reflection**: Uses language models to generate improved prompts
- **Merge Operations**: Combines successful prompts through crossover
- **Built-in Adapters**: Ready-to-use adapters for common tasks
- **Deterministic Testing**: Record/replay system for reproducible results

## Core Concepts

### Optimization Process

GEPA evolves prompts through these steps:

1. **Evaluate** current prompts on your training data
2. **Reflect** on failures using an LLM to generate improvements  
3. **Merge** successful prompts to create new variants
4. **Select** the best performers for the next iteration

### Adapters

Adapters connect GEPA to your specific task:

```typescript
import { BaseAdapter } from 'gepa-ts';

class SentimentAdapter extends BaseAdapter {
  async evaluate(batch, candidate, captureTraces = false) {
    const outputs = [];
    const scores = [];
    
    for (const example of batch) {
      const response = await this.callLLM(
        candidate.system_prompt + "\n" + example.input
      );
      const score = response.includes(example.answer) ? 1.0 : 0.0;
      
      outputs.push({ response });
      scores.push(score);
    }
    
    return { outputs, scores, trajectories: null };
  }
  
  async makeReflectiveDataset(candidate, evalBatch, componentsToUpdate) {
    // Generate examples for LLM reflection
    return {
      system_prompt: evalBatch.trajectories?.map(traj => ({
        'Input': traj.input,
        'Output': traj.output, 
        'Score': traj.score
      })) || []
    };
  }
}
```

## Built-in Adapters

### DefaultAdapter
Basic single-turn prompt evaluation:

```typescript
import { DefaultAdapter } from 'gepa-ts';

const adapter = new DefaultAdapter(async (prompt) => {
  return await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }]
  });
});
```

### Math Dataset
Optimizing mathematical problem solving:

```typescript
import { loadMathDataset } from 'gepa-ts';

const dataset = await loadMathDataset('algebra');
// Returns: { train: [...], dev: [...], test: [...] }

const result = await optimize({
  seedCandidate: { 
    system_prompt: "Solve this math problem step by step" 
  },
  trainset: dataset.train.slice(0, 10), // Start small
  maxMetricCalls: 100,
  reflectionLM: yourLLMFunction
});
```

## API Reference

### optimize(config)

Main optimization function with identical parameters to Python GEPA:

```typescript
interface OptimizeConfig {
  // Required
  seedCandidate: ComponentMap;           // Starting prompt(s)
  trainset: DataInst[];                  // Training examples
  maxMetricCalls: number;                // Evaluation budget
  
  // LLM Configuration  
  reflectionLM?: LanguageModel;          // LLM for generating improvements
  taskLM?: LanguageModel;                // LLM for task evaluation (if no adapter)
  
  // Optional
  valset?: DataInst[];                   // Validation set (defaults to trainset)
  adapter?: GEPAAdapter;                 // Custom evaluation adapter
  
  // Optimization Parameters
  reflectionMinibatchSize?: number;      // Examples per reflection (default: 3)
  useMerge?: boolean;                    // Enable crossover (default: false)
  maxMergeInvocations?: number;          // Max merges (default: 5)
  perfectScore?: number;                 // Early stopping score (default: 1.0)
  skipPerfectScore?: boolean;            // Skip if perfect (default: true)
  
  // Reproducibility
  seed?: number;                         // Random seed (default: 0)
  
  // Logging
  displayProgressBar?: boolean;          // Show progress (default: false)
  useWandB?: boolean;                   // Weights & Biases logging
}
```

### GEPAResult

Optimization results with complete Python compatibility:

```typescript
interface GEPAResult {
  // Core Results
  bestCandidate: ComponentMap;           // Best performing prompts
  bestScore: number;                     // Best validation score
  candidates: ComponentMap[];            // All generated candidates
  
  // Detailed Tracking
  valAggregateScores: number[];         // Per-candidate scores
  valSubscores: number[][];             // Per-instance scores
  perValInstanceBestCandidates: Set<number>[]; // Pareto front per example
  
  // Metadata
  totalEvaluations: number;             // Total LLM calls made
  generationsCompleted: number;         // Optimization iterations
  
  // Utility Methods
  nonDominatedIndices(): number[];      // Get Pareto optimal candidates
  lineage(idx: number): number[];       // Get candidate ancestry
  bestK(k: number): CandidateRanking[]; // Top k candidates
  toDict(): Record<string, any>;        // Serialize results
}
```

## Examples

### Sentiment Classification

```typescript
const result = await optimize({
  seedCandidate: { 
    instruction: "What is the sentiment?" 
  },
  trainset: [
    { text: "I love this!", sentiment: "positive" },
    { text: "This sucks", sentiment: "negative" }
  ],
  maxMetricCalls: 30,
  reflectionLM: async (prompt) => {
    return "Classify the sentiment of the given text as positive, negative, or neutral.";
  }
});
```

### Math Problem Solving

```typescript
const mathData = await loadMathDataset('algebra');

const result = await optimize({
  seedCandidate: {
    system_prompt: "You are a math tutor. Solve step by step."
  },
  trainset: mathData.train.slice(0, 20),
  maxMetricCalls: 100,
  useMerge: true,
  reflectionLM: async (prompt) => {
    return "You are a mathematical problem-solving assistant. Show your work clearly and provide the final numerical answer.";
  }
});
```

## Testing

Complete test coverage with Python parity validation:

```bash
npm test                    # Run all tests
npm test -- --run          # Skip watch mode  
npm run test:parity         # Validate Python compatibility
```

## Performance

- **Fast**: Native TypeScript performance with async/await
- **Memory efficient**: Lazy evaluation and proper garbage collection
- **Deterministic**: Reproducible results with seeded randomness
- **Feature complete**: 100% API compatibility with Python GEPA

## Development

```bash
git clone https://github.com/yourusername/gepa-ts
cd gepa-ts
npm install
npm run build
npm test
```

## License

MIT

## Citation

```bibtex
@software{gepa-ts,
  title = {GEPA-TS: TypeScript Implementation of GEPA},
  year = {2024},
  url = {https://github.com/yourusername/gepa-ts}
}
```

Based on the original Python implementation: [GEPA](https://github.com/gepa-ai/gepa)