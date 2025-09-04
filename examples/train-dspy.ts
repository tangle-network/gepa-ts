#!/usr/bin/env tsx
/**
 * DSPy training example - Structured prompt optimization
 * Demonstrates optimizing DSPy signatures and modules
 */

import { GEPAOptimizer } from '../src/core/optimizer.js';
import { DSPyAdapter } from '../src/adapters/dspy-adapter.js';
import { DSPyFullProgramAdapter } from '../src/adapters/dspy-full-program-adapter.js';
import { OpenAILanguageModel } from '../src/models/openai.js';
import { WandBIntegration } from '../src/integrations/wandb.js';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

// Example: Question Answering with Chain-of-Thought
interface QAExample {
  question: string;
  context: string;
  answer: string;
  reasoning?: string;
}

const QA_EXAMPLES: QAExample[] = [
  {
    question: "What is the capital of France?",
    context: "France is a country in Western Europe. Its capital and largest city is Paris, which is known for the Eiffel Tower.",
    answer: "Paris",
    reasoning: "The context states that Paris is the capital and largest city of France."
  },
  {
    question: "Who wrote Romeo and Juliet?",
    context: "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about two young star-crossed lovers.",
    answer: "William Shakespeare",
    reasoning: "The context directly states that Romeo and Juliet was written by William Shakespeare."
  },
  {
    question: "What is the speed of light?",
    context: "Light travels at approximately 299,792,458 meters per second in a vacuum, commonly approximated as 3×10^8 m/s.",
    answer: "299,792,458 meters per second",
    reasoning: "The context provides the exact speed of light in a vacuum."
  },
  {
    question: "What year did World War II end?",
    context: "World War II was a global war that lasted from 1939 to 1945. It ended with the surrender of Japan on September 2, 1945.",
    answer: "1945",
    reasoning: "The context indicates that World War II ended in 1945 with Japan's surrender."
  },
  {
    question: "What is the largest planet in our solar system?",
    context: "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun.",
    answer: "Jupiter",
    reasoning: "The context explicitly states that Jupiter is the largest planet in the Solar System."
  }
];

async function trainBasicDSPy() {
  console.log('🎯 Training Basic DSPy Signatures\n');
  
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY environment variable not set');
  }
  
  const lm = new OpenAILanguageModel({
    apiKey,
    model: 'gpt-3.5-turbo',
    temperature: 0.3,
    maxTokens: 200
  });
  
  // Create adapter for signature optimization
  const adapter = new DSPyAdapter({
    lm: async (prompt: string) => lm.generate(prompt),
    modules: {
      qa_module: {
        type: 'ChainOfThought',
        signature: {
          inputs: ['question', 'context'],
          outputs: ['reasoning', 'answer'],
          instruction: 'Answer the question based on the context. Think step by step.'
        }
      }
    }
  });
  
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 6,
      generations: 5,
      mutationRate: 0.3,
      mergeRate: 0.2,
      verbose: true,
      runDir: join(process.cwd(), 'runs', `dspy-basic-${Date.now()}`)
    }
  });
  
  // Initial DSPy signature components
  const initialComponents = {
    'qa_module.instruction': 'Answer the question based on the context. Think step by step.',
    'qa_module.input_format': 'Question: {question}\nContext: {context}',
    'qa_module.output_format': 'Reasoning: {reasoning}\nAnswer: {answer}',
    'qa_module.examples': JSON.stringify([
      {
        question: "What color is the sky?",
        context: "On a clear day, the sky appears blue due to Rayleigh scattering.",
        reasoning: "The context explains that the sky appears blue on clear days.",
        answer: "Blue"
      }
    ])
  };
  
  const trainData = QA_EXAMPLES.slice(0, 3);
  const valData = QA_EXAMPLES.slice(3);
  
  console.log('Optimizing QA module signature...');
  const result = await optimizer.optimize({
    initialCandidate: initialComponents,
    trainData,
    valData,
    componentsToUpdate: ['qa_module.instruction', 'qa_module.output_format']
  });
  
  console.log('\n✅ Basic DSPy Optimization Complete');
  console.log('Best score:', result.bestScore.toFixed(3));
  console.log('\nOptimized instruction:');
  console.log(result.bestCandidate['qa_module.instruction']);
  
  return result;
}

async function trainFullDSPyProgram() {
  console.log('\n🚀 Training Full DSPy Program Evolution\n');
  
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY environment variable not set');
  }
  
  const lm = new OpenAILanguageModel({
    apiKey,
    model: 'gpt-4-turbo-preview',
    temperature: 0.5,
    maxTokens: 500
  });
  
  // Create adapter for full program evolution
  const adapter = new DSPyFullProgramAdapter({
    lm: async (prompt: string) => lm.generate(prompt),
    allowedModuleTypes: ['Predict', 'ChainOfThought', 'ReAct'],
    maxModules: 3,
    evolutionStrategy: 'adaptive'
  });
  
  // Initialize WandB (optional)
  let wandb: WandBIntegration | undefined;
  if (process.env.WANDB_API_KEY) {
    wandb = new WandBIntegration({
      project: 'gepa-dspy',
      name: `dspy-program-evolution-${Date.now()}`,
      config: {
        dataset: 'QA',
        model: 'gpt-4-turbo-preview',
        evolutionStrategy: 'adaptive'
      }
    });
    await wandb.init();
  }
  
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 8,
      generations: 10,
      mutationRate: 0.4,
      mergeRate: 0.3,
      tournamentSize: 3,
      verbose: true,
      runDir: join(process.cwd(), 'runs', `dspy-full-${Date.now()}`)
    },
    wandbIntegration: wandb
  });
  
  // Initial DSPy program structure
  const initialProgram = {
    'program.structure': JSON.stringify({
      modules: [
        {
          name: 'understand',
          type: 'Predict',
          signature: {
            inputs: ['question', 'context'],
            outputs: ['key_facts'],
            instruction: 'Extract key facts from the context relevant to the question'
          }
        },
        {
          name: 'reason',
          type: 'ChainOfThought',
          signature: {
            inputs: ['question', 'key_facts'],
            outputs: ['reasoning', 'answer'],
            instruction: 'Use the key facts to reason through the question step by step'
          }
        }
      ],
      flow: 'understand -> reason'
    }),
    'program.optimization': JSON.stringify({
      focus: 'accuracy',
      constraints: {
        maxLatency: 1000,
        maxTokens: 500
      }
    })
  };
  
  // More complex QA examples for program evolution
  const complexQAExamples = [
    {
      question: "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
      context: "Each machine works independently and at the same rate. The production rate is constant.",
      answer: "5 minutes",
      reasoning: "Each machine makes 1 widget in 5 minutes. With 100 machines working in parallel, they can make 100 widgets in the same 5 minutes."
    },
    {
      question: "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
      context: "This appears to be a mathematical sequence. Look for patterns in the differences or formulas.",
      answer: "42",
      reasoning: "The differences between consecutive terms are 4, 6, 8, 10, which increases by 2 each time. Next difference would be 12, so 30 + 12 = 42."
    },
    {
      question: "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
      context: "This is a logical reasoning problem. Consider the relationships between the sets.",
      answer: "No",
      reasoning: "We cannot conclude this. While all roses are flowers, the flowers that fade quickly might not include any roses."
    }
  ];
  
  const trainData = complexQAExamples.slice(0, 2);
  const valData = complexQAExamples.slice(2);
  
  console.log('Evolving DSPy program structure...');
  const result = await optimizer.optimize({
    initialCandidate: initialProgram,
    trainData,
    valData,
    componentsToUpdate: ['program.structure']
  });
  
  console.log('\n✅ Full DSPy Program Evolution Complete');
  console.log('Best score:', result.bestScore.toFixed(3));
  
  // Parse and display evolved program
  try {
    const evolvedProgram = JSON.parse(result.bestCandidate['program.structure']);
    console.log('\nEvolved Program Structure:');
    console.log('Modules:', evolvedProgram.modules.length);
    for (const module of evolvedProgram.modules) {
      console.log(`  - ${module.name} (${module.type})`);
    }
    console.log('Flow:', evolvedProgram.flow);
  } catch (e) {
    console.log('Could not parse evolved program structure');
  }
  
  // Close WandB
  if (wandb) {
    await wandb.finish();
  }
  
  return result;
}

async function main() {
  console.log('🧠 DSPy Optimization Examples\n');
  console.log('This example demonstrates two approaches:');
  console.log('1. Basic: Optimizing DSPy signatures and instructions');
  console.log('2. Advanced: Evolving entire DSPy program structures\n');
  
  const outputDir = join(process.cwd(), 'results', 'dspy');
  mkdirSync(outputDir, { recursive: true });
  
  try {
    // Run basic signature optimization
    const basicResult = await trainBasicDSPy();
    
    // Save basic results
    writeFileSync(
      join(outputDir, 'basic_optimization.json'),
      JSON.stringify(basicResult, null, 2)
    );
    
    // Run full program evolution
    const fullResult = await trainFullDSPyProgram();
    
    // Save full results
    writeFileSync(
      join(outputDir, 'program_evolution.json'),
      JSON.stringify(fullResult, null, 2)
    );
    
    console.log('\n🎉 All DSPy optimizations complete!');
    console.log(`Results saved to: ${outputDir}`);
    
  } catch (error) {
    console.error('❌ Error during optimization:', error);
    process.exit(1);
  }
}

// Run the example
main().catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});