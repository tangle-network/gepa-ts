/**
 * AIME Dataset integration for mathematical reasoning tasks
 * Loads AIME (American Invitational Mathematics Examination) problems
 */

import { MathProblem } from '../adapters/anymaths-adapter.js';

export interface AIMEProblem extends MathProblem {
  year?: number;
  problemNumber?: number;
  source: string;
  topic?: string;
  difficulty: 'medium' | 'hard';
}

export interface AIMEDatasetConfig {
  split?: 'train' | 'validation' | 'test';
  maxProblems?: number;
  shuffle?: boolean;
  seed?: number;
}

export class AIMEDataset {
  private problems: AIMEProblem[] = [];
  
  constructor(private config: AIMEDatasetConfig = {}) {
    this.config = {
      split: 'train',
      shuffle: true,
      seed: 0,
      ...config
    };
  }
  
  /**
   * Load AIME dataset from HuggingFace or local cache
   */
  async load(): Promise<void> {
    try {
      // In a real implementation, this would load from HuggingFace datasets
      // For now, we'll use mock data
      this.problems = this.getMockAIMEProblems();
      
      if (this.config.shuffle) {
        this.shuffle();
      }
      
      if (this.config.maxProblems) {
        this.problems = this.problems.slice(0, this.config.maxProblems);
      }
    } catch (error) {
      console.error('Failed to load AIME dataset:', error);
      // Fallback to mock data
      this.problems = this.getMockAIMEProblems();
    }
  }
  
  /**
   * Get mock AIME problems for testing
   */
  private getMockAIMEProblems(): AIMEProblem[] {
    return [
      {
        problem: "What is the slope of a line perpendicular to the line whose equation is $\\frac{x}{4}-\\frac{y}{5}=1$? Express your answer as a common fraction.",
        answer: "-4/5",
        difficulty: 'medium',
        source: 'AIME',
        topic: 'algebra'
      },
      {
        problem: "In the equation $|x-4| -10 = 2$, what is the product of all possible values of $x$?",
        answer: "-128",
        difficulty: 'medium',
        source: 'AIME',
        topic: 'algebra'
      },
      {
        problem: "If $f(x) = x^3 - 6x^2 + 3x - 4$, $g(x) = x^3 + 5x^2 + 9x - 2$, then find the constant term of $f(g(x))$.",
        answer: "-42",
        difficulty: 'hard',
        source: 'AIME',
        topic: 'functions'
      },
      {
        problem: "Find the number of ordered pairs $(x,y)$ of positive integers such that $\\text{lcm}(x,y) = 120$ and $\\gcd(x,y) = 4$.",
        answer: "10",
        difficulty: 'hard',
        source: 'AIME',
        topic: 'number_theory'
      },
      {
        problem: "A circle with center $(3, -2)$ is tangent to the line $3x - 4y = 17$. What is the radius of the circle?",
        answer: "2",
        difficulty: 'medium',
        source: 'AIME',
        topic: 'geometry'
      }
    ];
  }
  
  /**
   * Shuffle problems using seeded random
   */
  private shuffle(): void {
    const rng = this.createRNG(this.config.seed!);
    
    for (let i = this.problems.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [this.problems[i], this.problems[j]] = [this.problems[j], this.problems[i]];
    }
  }
  
  /**
   * Create seeded random number generator
   */
  private createRNG(seed: number): () => number {
    let s = seed;
    return () => {
      s = Math.sin(s) * 10000;
      return s - Math.floor(s);
    };
  }
  
  /**
   * Split dataset into train/val/test
   */
  split(trainRatio: number = 0.6, valRatio: number = 0.2): {
    train: AIMEProblem[];
    val: AIMEProblem[];
    test: AIMEProblem[];
  } {
    const n = this.problems.length;
    const trainEnd = Math.floor(n * trainRatio);
    const valEnd = Math.floor(n * (trainRatio + valRatio));
    
    return {
      train: this.problems.slice(0, trainEnd),
      val: this.problems.slice(trainEnd, valEnd),
      test: this.problems.slice(valEnd)
    };
  }
  
  /**
   * Get all problems
   */
  getProblems(): AIMEProblem[] {
    return this.problems;
  }
  
  /**
   * Get problems by topic
   */
  getProblemsByTopic(topic: string): AIMEProblem[] {
    return this.problems.filter(p => p.topic === topic);
  }
  
  /**
   * Get problems by difficulty
   */
  getProblemsByDifficulty(difficulty: 'medium' | 'hard'): AIMEProblem[] {
    return this.problems.filter(p => p.difficulty === difficulty);
  }
  
  /**
   * Convert to format expected by GEPA
   */
  toGEPAFormat(): Array<{
    input: string;
    answer: string;
    additional_context?: any;
  }> {
    return this.problems.map(p => ({
      input: p.problem,
      answer: `### ${p.answer}`,
      additional_context: {
        topic: p.topic,
        difficulty: p.difficulty,
        solution: p.solution
      }
    }));
  }
}

/**
 * Initialize AIME dataset (matches Python API)
 */
export async function initDataset(config?: AIMEDatasetConfig): Promise<{
  trainset: Array<any>;
  valset: Array<any>;
  testset: Array<any>;
}> {
  const dataset = new AIMEDataset(config);
  await dataset.load();
  
  const splits = dataset.split();
  
  // Convert to GEPA format
  const toGEPA = (problems: AIMEProblem[]) => problems.map(p => ({
    input: p.problem,
    answer: `### ${p.answer}`,
    additional_context: {
      topic: p.topic,
      difficulty: p.difficulty
    }
  }));
  
  return {
    trainset: toGEPA(splits.train),
    valset: toGEPA(splits.val),
    testset: toGEPA(splits.test)
  };
}

/**
 * Load AIME 2025 dataset specifically
 */
export async function loadAIME2025(): Promise<AIMEProblem[]> {
  // In production, this would fetch from HuggingFace: MathArena/aime_2025
  const problems: AIMEProblem[] = [
    {
      problem: "Suppose that $a$, $b$, and $c$ are positive real numbers such that $a + b + c = 10$ and $ab + bc + ca = 25$. Find $abc$.",
      answer: "50/3",
      year: 2025,
      problemNumber: 1,
      difficulty: 'hard',
      source: 'AIME 2025',
      topic: 'algebra'
    },
    // Add more 2025 problems as they become available
  ];
  
  return problems;
}

/**
 * Load validation split from AI-MO/aimo-validation-aime
 */
export async function loadValidationSplit(): Promise<AIMEProblem[]> {
  // In production, this would fetch from HuggingFace: AI-MO/aimo-validation-aime
  const dataset = new AIMEDataset({ split: 'validation' });
  await dataset.load();
  return dataset.getProblems();
}