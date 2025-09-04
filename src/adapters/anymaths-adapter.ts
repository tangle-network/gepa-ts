import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, DataInst, ReflectiveDataset, RolloutOutput, Trajectory } from '../types/index.js';

/**
 * AnyMaths Adapter for mathematical problem solving optimization
 * Specializes in optimizing prompts for mathematical reasoning
 */

export interface MathProblem extends DataInst {
  problem: string;
  answer: string | number;
  difficulty?: 'easy' | 'medium' | 'hard';
  topic?: string;
  solution?: string; // Step-by-step solution if available
}

export interface MathSolutionTrace extends Trajectory {
  problem: string;
  steps: Array<{
    step: number;
    description: string;
    calculation: string;
    result: string | number;
  }>;
  finalAnswer: string | number;
  isCorrect: boolean;
  errors?: string[];
}

export interface MathSolutionOutput extends RolloutOutput {
  answer: string | number;
  reasoning?: string;
  steps?: string[];
  confidence?: number;
  trace?: MathSolutionTrace;
}

export interface AnyMathsAdapterConfig {
  model: string | ((prompt: string) => Promise<string> | string);
  extractAnswer?: (response: string) => string | number;
  checkAnswer?: (predicted: string | number, expected: string | number) => boolean;
  requireSteps?: boolean;
  timeout?: number;
}

export class AnyMathsAdapter extends BaseAdapter<MathProblem, MathSolutionTrace, MathSolutionOutput> {
  private config: AnyMathsAdapterConfig;
  
  constructor(config: AnyMathsAdapterConfig) {
    super();
    this.config = {
      requireSteps: true,
      timeout: 30000,
      extractAnswer: this.defaultAnswerExtractor,
      checkAnswer: this.defaultAnswerChecker,
      ...config
    };
  }
  
  private defaultAnswerExtractor(response: string): string | number {
    // Try to extract answer from various formats
    const patterns = [
      /(?:answer|solution|result)[\s:]*([+-]?\d+\.?\d*)/i,
      /\$?([+-]?\d+\.?\d*)\$?$/,
      /therefore,?\s+([+-]?\d+\.?\d*)/i,
      /=\s*([+-]?\d+\.?\d*)(?:\s|$)/,
      /\boxed\{([^}]+)\}/,
      /###\s*([+-]?\d+\.?\d*)/
    ];
    
    for (const pattern of patterns) {
      const match = response.match(pattern);
      if (match) {
        const value = match[1].trim();
        // Try to parse as number
        const num = parseFloat(value);
        if (!isNaN(num)) {
          return num;
        }
        return value;
      }
    }
    
    // Fallback: look for last number in response
    const numbers = response.match(/[+-]?\d+\.?\d*/g);
    if (numbers && numbers.length > 0) {
      const lastNum = parseFloat(numbers[numbers.length - 1]);
      if (!isNaN(lastNum)) {
        return lastNum;
      }
    }
    
    return '';
  }
  
  private defaultAnswerChecker(predicted: string | number, expected: string | number): boolean {
    // Convert to numbers if possible
    const predNum = typeof predicted === 'number' ? predicted : parseFloat(String(predicted));
    const expNum = typeof expected === 'number' ? expected : parseFloat(String(expected));
    
    if (!isNaN(predNum) && !isNaN(expNum)) {
      // Numerical comparison with tolerance
      return Math.abs(predNum - expNum) < 0.0001;
    }
    
    // String comparison (normalize fractions, etc.)
    const predStr = String(predicted).trim().toLowerCase();
    const expStr = String(expected).trim().toLowerCase();
    
    // Handle fractions
    if (predStr.includes('/') || expStr.includes('/')) {
      const evalFraction = (frac: string): number => {
        const parts = frac.split('/');
        if (parts.length === 2) {
          return parseFloat(parts[0]) / parseFloat(parts[1]);
        }
        return parseFloat(frac);
      };
      
      const predVal = evalFraction(predStr);
      const expVal = evalFraction(expStr);
      
      if (!isNaN(predVal) && !isNaN(expVal)) {
        return Math.abs(predVal - expVal) < 0.0001;
      }
    }
    
    return predStr === expStr;
  }
  
  private async solveProblem(
    problem: string,
    systemPrompt: string,
    captureTrace: boolean = false
  ): Promise<MathSolutionOutput> {
    const fullPrompt = `${systemPrompt}\n\nProblem: ${problem}`;
    
    try {
      let response: string;
      
      if (typeof this.config.model === 'function') {
        response = await this.config.model(fullPrompt);
      } else {
        // Mock response for testing
        response = `Let me solve this step by step.\n\nStep 1: Analyze the problem\nStep 2: Apply formula\nStep 3: Calculate\n\nThe answer is 42.`;
      }
      
      // Extract answer
      const answer = this.config.extractAnswer!(response);
      
      // Extract reasoning and steps
      const steps = this.extractSteps(response);
      const reasoning = this.extractReasoning(response);
      
      // Build trace if needed
      let trace: MathSolutionTrace | undefined;
      if (captureTrace) {
        trace = {
          problem,
          steps: steps.map((step, i) => ({
            step: i + 1,
            description: step,
            calculation: '',
            result: i === steps.length - 1 ? answer : ''
          })),
          finalAnswer: answer,
          isCorrect: false // Will be set during evaluation
        };
      }
      
      return {
        answer,
        reasoning,
        steps,
        confidence: this.calculateConfidence(response),
        trace
      };
      
    } catch (error) {
      return {
        answer: '',
        reasoning: `Error: ${error}`,
        steps: [],
        confidence: 0,
        trace: captureTrace ? {
          problem,
          steps: [],
          finalAnswer: '',
          isCorrect: false,
          errors: [String(error)]
        } : undefined
      };
    }
  }
  
  private extractSteps(response: string): string[] {
    const steps: string[] = [];
    
    // Look for numbered steps
    const stepPattern = /(?:step\s*\d+|^\d+\.|\d+\))\s*(.+?)(?=(?:step\s*\d+|^\d+\.|\d+\)|$))/gmi;
    let match;
    
    while ((match = stepPattern.exec(response)) !== null) {
      steps.push(match[1].trim());
    }
    
    // If no numbered steps, try to split by newlines
    if (steps.length === 0) {
      const lines = response.split('\n').filter(line => 
        line.trim() && 
        !line.startsWith('Problem:') &&
        !line.startsWith('Answer:')
      );
      return lines.slice(0, 5); // Limit to 5 steps
    }
    
    return steps;
  }
  
  private extractReasoning(response: string): string {
    // Extract the main reasoning part
    const reasoningPattern = /(?:reasoning|solution|approach):?\s*([\s\S]+?)(?:answer|result|therefore|$)/i;
    const match = response.match(reasoningPattern);
    
    if (match) {
      return match[1].trim();
    }
    
    // Fallback: use the first part of response
    return response.split('\n').slice(0, 3).join('\n');
  }
  
  private calculateConfidence(response: string): number {
    let confidence = 0.5;
    
    // Higher confidence if response includes steps
    if (response.includes('Step') || response.includes('step')) {
      confidence += 0.2;
    }
    
    // Higher confidence if response includes verification
    if (response.includes('verify') || response.includes('check')) {
      confidence += 0.1;
    }
    
    // Higher confidence if response is detailed
    if (response.length > 200) {
      confidence += 0.1;
    }
    
    // Lower confidence if response includes uncertainty
    if (response.includes('might') || response.includes('possibly') || response.includes('uncertain')) {
      confidence -= 0.2;
    }
    
    return Math.max(0, Math.min(1, confidence));
  }
  
  async evaluate(
    batch: MathProblem[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<MathSolutionTrace, MathSolutionOutput>> {
    const systemPrompt = candidate['math_prompt'] || candidate['system_prompt'] || 
      'You are a mathematical problem solver. Show your work step by step.';
    
    const outputs: MathSolutionOutput[] = [];
    const scores: number[] = [];
    const trajectories: MathSolutionTrace[] | null = captureTraces ? [] : null;
    
    for (const problem of batch) {
      const solution = await this.solveProblem(
        problem.problem,
        systemPrompt,
        captureTraces
      );
      
      // Check correctness
      const isCorrect = this.config.checkAnswer!(solution.answer, problem.answer);
      const score = isCorrect ? 1.0 : 0.0;
      
      // Update trace
      if (solution.trace) {
        solution.trace.isCorrect = isCorrect;
      }
      
      outputs.push(solution);
      scores.push(score);
      
      if (trajectories && solution.trace) {
        trajectories.push(solution.trace);
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<MathSolutionTrace, MathSolutionOutput>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    if (!evalBatch.trajectories) {
      return dataset;
    }
    
    for (const componentName of componentsToUpdate) {
      const examples: Array<Record<string, any>> = [];
      
      for (let i = 0; i < evalBatch.trajectories.length; i++) {
        const trace = evalBatch.trajectories[i];
        const output = evalBatch.outputs[i];
        const score = evalBatch.scores[i];
        
        let feedback = '';
        let improvements: string[] = [];
        
        if (!trace.isCorrect) {
          feedback = `Incorrect answer. Expected ${trace.problem}, got ${output.answer}. `;
          
          // Analyze error patterns
          if (!output.reasoning || output.reasoning.length < 50) {
            improvements.push('Show more detailed work');
          }
          
          if (!output.steps || output.steps.length < 2) {
            improvements.push('Break down problem into clear steps');
          }
          
          if (trace.errors && trace.errors.length > 0) {
            feedback += `Errors: ${trace.errors.join(', ')}. `;
            improvements.push('Check calculations carefully');
          }
          
          // Topic-specific feedback
          if (trace.problem.includes('fraction')) {
            improvements.push('Be careful with fraction operations');
          }
          if (trace.problem.includes('percent')) {
            improvements.push('Convert percentages correctly');
          }
          if (trace.problem.includes('equation')) {
            improvements.push('Isolate variables systematically');
          }
          
        } else {
          feedback = 'Correct answer. ';
          
          if (output.confidence && output.confidence > 0.8) {
            feedback += 'High confidence solution. ';
          }
          
          if (output.steps && output.steps.length > 5) {
            improvements.push('Consider if solution can be simplified');
          }
        }
        
        examples.push({
          'Problem': trace.problem,
          'Generated Answer': output.answer,
          'Correct Answer': evalBatch.trajectories[i].finalAnswer,
          'Score': score,
          'Feedback': feedback,
          'Improvements': improvements,
          'Solution Steps': output.steps || [],
          'Current Prompt': candidate[componentName] || ''
        });
      }
      
      dataset[componentName] = examples.slice(0, 5);
    }
    
    return dataset;
  }
  
  proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): ComponentMap {
    const newTexts: ComponentMap = {};
    
    for (const componentName of componentsToUpdate) {
      const examples = reflectiveDataset[componentName] || [];
      let currentPrompt = candidate[componentName] || '';
      
      // Analyze error patterns
      const allImprovements = examples.flatMap(ex => ex['Improvements'] || []);
      const incorrectExamples = examples.filter(ex => ex['Score'] < 0.5);
      const correctExamples = examples.filter(ex => ex['Score'] >= 1.0);
      
      // Build improved prompt
      let improvedPrompt = currentPrompt;
      
      // Add step-by-step guidance if missing
      if (!improvedPrompt.includes('step') && incorrectExamples.length > 0) {
        improvedPrompt = 'Solve mathematical problems step by step. ' + improvedPrompt;
      }
      
      // Add verification step
      if (!improvedPrompt.includes('verify') && incorrectExamples.length > correctExamples.length) {
        improvedPrompt += '\n\nAlways verify your answer by checking it against the problem.';
      }
      
      // Add specific guidance based on errors
      const uniqueImprovements = [...new Set(allImprovements)];
      if (uniqueImprovements.length > 0) {
        improvedPrompt += '\n\nKey points to remember:\n';
        improvedPrompt += uniqueImprovements.slice(0, 5).map(imp => `- ${imp}`).join('\n');
      }
      
      // Add format specification
      if (!improvedPrompt.includes('answer format')) {
        improvedPrompt += '\n\nProvide your final answer clearly marked, e.g., "Answer: [your answer]"';
      }
      
      // Add examples of good solutions if available
      if (correctExamples.length > 0) {
        const goodSteps = correctExamples[0]['Solution Steps'];
        if (goodSteps && goodSteps.length > 0) {
          improvedPrompt += '\n\nExample of good solution structure:\n';
          improvedPrompt += goodSteps.slice(0, 3).map((s: string, i: number) => 
            `Step ${i + 1}: ${s}`
          ).join('\n');
        }
      }
      
      newTexts[componentName] = improvedPrompt;
    }
    
    return newTexts;
  }
}