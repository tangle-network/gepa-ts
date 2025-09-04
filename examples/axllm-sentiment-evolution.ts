/**
 * AxLLM + GEPA: Evolving Sentiment Analysis Programs
 * 
 * This example shows how GEPA can evolve AxLLM programs beyond what
 * MiPRO alone can achieve by:
 * 1. Evolving the signature structure itself
 * 2. Selecting optimal demos through genetic algorithm
 * 3. Tuning model configs across generations
 * 4. Combining multiple optimization strategies
 */

import { ax, ai, AxOptimizedProgramImpl } from '@ax-llm/ax';
import { GEPAOptimizer } from '../src/core/optimizer.js';
import { AxLLMNativeAdapter, createAxLLMNativeAdapter } from '../src/adapters/axllm-native-adapter.js';
import { promises as fs } from 'fs';

async function main() {
  console.log('🧬 GEPA + AxLLM: Evolving Sentiment Analysis Beyond MiPRO\n');

  // Step 1: Set up AxLLM
  const llm = ai({
    name: 'openai',
    apiKey: process.env.OPENAI_API_KEY!,
    config: { model: 'gpt-4o-mini' }
  });

  // Step 2: Define initial program (simple, will be evolved)
  const initialProgram = {
    signature: 'reviewText:string "Customer review" -> sentiment:class "positive,negative,neutral" "Sentiment"',
    demos: [], // Start with no demos
    modelConfig: {
      temperature: 0.7,
      maxTokens: 50
    }
  };

  // Step 3: Training data (same as MiPRO example)
  const trainingExamples = [
    { reviewText: "I love this product!", sentiment: "positive" },
    { reviewText: "This is terrible quality", sentiment: "negative" },
    { reviewText: "It works fine, nothing special", sentiment: "neutral" },
    { reviewText: "Best purchase ever!", sentiment: "positive" },
    { reviewText: "Waste of money", sentiment: "negative" },
    { reviewText: "The product is okay", sentiment: "neutral" },
    { reviewText: "Exceeded my expectations", sentiment: "positive" },
    { reviewText: "Broke after one day", sentiment: "negative" },
    { reviewText: "Does what it says", sentiment: "neutral" },
    { reviewText: "Absolutely fantastic!", sentiment: "positive" }
  ];

  // Validation set (different from training)
  const validationExamples = [
    { reviewText: "Amazing quality and fast shipping", sentiment: "positive" },
    { reviewText: "Completely disappointed", sentiment: "negative" },
    { reviewText: "It's adequate for the price", sentiment: "neutral" },
    { reviewText: "Would buy again!", sentiment: "positive" },
    { reviewText: "Don't waste your money", sentiment: "negative" }
  ];

  // Step 4: Define metric (same as MiPRO)
  const metric = ({ prediction, example }: { prediction: any; example: any }) => {
    return prediction.sentiment === example.sentiment ? 1 : 0;
  };

  // Step 5: Create GEPA adapter for AxLLM
  const adapter = createAxLLMNativeAdapter({
    axInstance: ax,
    llm,
    baseProgram: initialProgram,
    metricFn: metric,
    reflectionLm: async (prompt: string) => {
      // Use GPT-4 for program evolution proposals
      const response = await llm.generate(prompt);
      return response;
    }
  });

  // Step 6: Run GEPA optimization
  console.log('🔄 Starting GEPA evolution (this goes beyond MiPRO)...\n');
  
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 10,     // 10 different program variants
      generations: 5,         // 5 evolutionary generations
      mutationRate: 0.3,      // 30% mutation rate
      crossoverRate: 0.5,     // 50% crossover rate
      elitismRate: 0.2,       // Keep top 20% of programs
      verbose: true
    }
  });

  const startTime = Date.now();
  
  const result = await optimizer.optimize({
    initialCandidate: { program: JSON.stringify(initialProgram) },
    trainData: trainingExamples,
    valData: validationExamples,
    componentsToUpdate: ['program'],
    maxMetricCalls: 100
  });

  const optimizationTime = Date.now() - startTime;

  // Step 7: Extract and display results
  const bestProgram = JSON.parse(result.bestCandidate.program);
  const bestScore = result.bestScore;

  console.log('\n✅ GEPA Evolution Complete!');
  console.log(`   Score: ${(bestScore * 100).toFixed(1)}% accuracy`);
  console.log(`   Time: ${(optimizationTime / 1000).toFixed(1)}s`);
  console.log(`   Generations: ${result.generation}`);
  console.log(`   Total evaluations: ${result.evaluationCount}`);

  // Step 8: Show evolution path
  console.log('\n📊 Evolution Path:');
  console.log('   Initial: Simple signature, no demos');
  console.log(`   Final: ${bestProgram.demos?.length || 0} demos selected`);
  console.log(`   Temperature: ${initialProgram.modelConfig.temperature} → ${bestProgram.modelConfig?.temperature || 0.7}`);
  
  if (bestProgram.signature !== initialProgram.signature) {
    console.log('   Signature: EVOLVED! ✨');
  }

  // Step 9: Compare with MiPRO approach
  console.log('\n🔬 GEPA vs MiPRO Advantages:');
  console.log('   1. Population-based: Tests multiple variants in parallel');
  console.log('   2. Genetic crossover: Combines successful features');
  console.log('   3. Signature evolution: Can modify the program structure');
  console.log('   4. Global optimization: Less likely to get stuck in local optima');
  console.log('   5. Pareto frontier: Tracks multiple good solutions');

  // Step 10: Save as AxOptimizedProgram format
  const axOptimized = adapter.toAxOptimizedProgram(
    result.bestCandidate,
    bestScore,
    {
      rounds: result.generation,
      evaluations: result.evaluationCount,
      populationDiversity: result.candidates.length
    }
  );

  await fs.writeFile(
    'sentiment-gepa-optimized.json',
    JSON.stringify({
      version: '3.0',
      optimizer: 'GEPA',
      comparisonWithMiPRO: {
        gepaScore: bestScore,
        gepaTime: optimizationTime,
        advantages: [
          'Population diversity',
          'Genetic crossover',
          'Structure evolution',
          'Global search'
        ]
      },
      ...axOptimized,
      evolutionHistory: result.history?.map((gen: any) => ({
        generation: gen.generation,
        bestScore: gen.bestScore,
        avgScore: gen.avgScore,
        diversity: gen.candidates?.length || 0
      }))
    }, null, 2)
  );

  console.log('\n💾 Optimization saved to sentiment-gepa-optimized.json');

  // Step 11: Production usage example
  console.log('\n🚀 Production Usage Example:\n');
  console.log(`
// Load GEPA-optimized program
const savedOptimization = JSON.parse(
  await fs.readFile('sentiment-gepa-optimized.json', 'utf8')
);

// Create AxLLM program
const sentimentAnalyzer = ax(savedOptimization.instruction);

// Apply GEPA optimization
const optimizedProgram = new AxOptimizedProgramImpl(savedOptimization);
sentimentAnalyzer.applyOptimization(optimizedProgram);

// Use in production with GEPA-evolved performance!
const result = await sentimentAnalyzer.forward(llm, {
  reviewText: "The product quality exceeded my expectations!"
});

console.log(\`Sentiment: \${result.sentiment}\`); // Highly accurate!
`);

  // Step 12: Show actual usage
  console.log('📝 Testing evolved program on new review:');
  const evolvedAnalyzer = ax(bestProgram.signature);
  
  if (bestProgram.demos) {
    evolvedAnalyzer.setDemos(bestProgram.demos);
  }
  
  const testResult = await evolvedAnalyzer.forward(llm, {
    reviewText: "The product arrived quickly but the quality was disappointing"
  });
  
  console.log(`   Review: "The product arrived quickly but the quality was disappointing"`);
  console.log(`   Predicted: ${testResult.sentiment}`);
  console.log(`   Expected: negative or neutral`);
  console.log(`   Correct: ${['negative', 'neutral'].includes(testResult.sentiment) ? '✅' : '❌'}`);
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { main };