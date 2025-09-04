#!/usr/bin/env node

import { optimize } from '../dist/index.js';
import { SilentLogger } from '../dist/logging/logger.js';

// Simple math problem dataset
const trainset = [
  { question: "What is 2 + 2?", answer: "4" },
  { question: "What is 5 * 3?", answer: "15" },
  { question: "What is 10 - 7?", answer: "3" },
  { question: "What is 20 / 4?", answer: "5" },
  { question: "What is 8 + 6?", answer: "14" }
];

const valset = [
  { question: "What is 3 + 4?", answer: "7" },
  { question: "What is 6 * 2?", answer: "12" },
  { question: "What is 15 - 9?", answer: "6" }
];

// Seed candidate with basic prompt
const seedCandidate = {
  system_prompt: "You are a helpful assistant. Answer the math question."
};

// Mock language model for testing
function mockTaskLM(prompt: string): string {
  // Simple regex-based math solver for testing
  const match = prompt.match(/(\d+)\s*([\+\-\*\/])\s*(\d+)/);
  if (match) {
    const a = parseInt(match[1]);
    const op = match[2];
    const b = parseInt(match[3]);
    
    let result: number;
    switch(op) {
      case '+': result = a + b; break;
      case '-': result = a - b; break;
      case '*': result = a * b; break;
      case '/': result = a / b; break;
      default: return "Error";
    }
    
    return result.toString();
  }
  return "Error";
}

// Mock reflection LM for testing
function mockReflectionLM(prompt: string): string {
  // Simple improvement: add more structure to the prompt
  if (prompt.includes("system_prompt")) {
    return "You are a helpful math assistant. Please solve the following math problem step by step and provide only the final numeric answer.";
  }
  return "Improved prompt";
}

async function runExample() {
  console.log("Running GEPA optimization example...");
  console.log("Initial seed candidate:", seedCandidate);
  console.log("\nStarting optimization...\n");
  
  try {
    const result = await optimize({
      seedCandidate,
      trainset,
      valset,
      taskLM: mockTaskLM,
      reflectionLM: mockReflectionLM,
      maxMetricCalls: 50,
      candidateSelectionStrategy: 'pareto',
      reflectionMinibatchSize: 2,
      perfectScore: 1,
      skipPerfectScore: true,
      useMerge: false,
      logger: new SilentLogger(),
      runDir: null,
      seed: 42,
      displayProgressBar: false
    });
    
    console.log("\nOptimization complete!");
    console.log("\nResults:");
    console.log("- Best score:", result.bestScore.toFixed(3));
    console.log("- Score improvement:", (result.bestScore - result.allScores[0]).toFixed(3));
    console.log("- Total evaluations:", result.totalEvaluations);
    console.log("- Candidates evaluated:", result.allCandidates.length);
    console.log("\nBest candidate found:");
    console.log(JSON.stringify(result.bestCandidate, null, 2));
    
  } catch (error) {
    console.error("Error during optimization:", error);
  }
}

runExample();