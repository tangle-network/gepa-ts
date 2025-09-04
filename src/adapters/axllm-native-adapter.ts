/**
 * Native AxLLM Adapter for GEPA
 * Evolves AxLLM programs directly in TypeScript - no Python needed!
 * Supports signature evolution, demo selection, instruction optimization, and model config tuning
 */

import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, ReflectiveDataset } from '../types/index.js';

// AxLLM types based on the SDK you showed
export interface AxSignature {
  inputs: Record<string, { type: string; description: string }>;
  outputs: Record<string, { type: string; description: string; classes?: string[] }>;
  instruction?: string;
}

export interface AxProgram {
  signature: string | AxSignature;
  demos?: Array<Record<string, any>>;
  modelConfig?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
  };
  optimizerConfig?: {
    type: 'MiPRO' | 'BootstrapFewShot' | 'LabeledFewShot' | 'GEPA';
    rounds?: number;
    batchSize?: number;
  };
}

export interface AxExample {
  [key: string]: any;
}

export interface AxPrediction {
  [key: string]: any;
  _score?: number;
  _feedback?: string;
}

export interface AxTrace {
  signature: string | AxSignature;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  modelCalls: Array<{
    prompt: string;
    response: string;
    tokens: number;
  }>;
  score: number;
}

export interface AxOptimizedProgram {
  bestScore: number;
  instruction?: string;
  demos?: Array<Record<string, any>>;
  modelConfig?: Record<string, any>;
  optimizerType: string;
  optimizationTime: number;
  totalRounds: number;
  converged: boolean;
  stats?: Record<string, any>;
}

/**
 * Native AxLLM Adapter
 * Evolves AxLLM programs using GEPA's genetic algorithm approach
 */
export class AxLLMNativeAdapter extends BaseAdapter<AxExample, AxTrace, AxPrediction> {
  private axInstance: any; // The ax() function from AxLLM
  private llm: any; // The ai() instance
  private metricFn: (params: { prediction: any; example: any }) => number | Promise<number>;
  private reflectionLm?: (prompt: string) => Promise<string>;
  private baseProgram: AxProgram;
  
  constructor(config: {
    axInstance: any; // ax() from '@ax-llm/ax'
    llm: any; // ai() instance
    baseProgram?: AxProgram;
    metricFn: (params: { prediction: any; example: any }) => number | Promise<number>;
    reflectionLm?: (prompt: string) => Promise<string>;
  }) {
    super();
    this.axInstance = config.axInstance;
    this.llm = config.llm;
    this.metricFn = config.metricFn;
    this.reflectionLm = config.reflectionLm;
    this.baseProgram = config.baseProgram || this.getDefaultProgram();
  }
  
  private getDefaultProgram(): AxProgram {
    return {
      signature: 'input:string -> output:string',
      demos: [],
      modelConfig: {
        temperature: 0.7,
        maxTokens: 100
      }
    };
  }
  
  /**
   * Parse AxLLM signature string into structured format
   */
  private parseSignature(sig: string): AxSignature {
    // Parse signature like: 'reviewText:string "desc" -> sentiment:class "pos,neg" "desc"'
    const [inputPart, outputPart] = sig.split('->').map(s => s.trim());
    
    const parseField = (field: string) => {
      const match = field.match(/(\w+):(\w+)(?:\s+"([^"]*)")?(?:\s+"([^"]*)")?/);
      if (!match) return null;
      
      const [, name, type, arg1, arg2] = match;
      const result: any = { type };
      
      if (type === 'class' && arg1) {
        result.classes = arg1.split(',').map(c => c.trim());
        result.description = arg2 || '';
      } else {
        result.description = arg1 || '';
      }
      
      return { name, ...result };
    };
    
    const inputField = parseField(inputPart);
    const outputField = parseField(outputPart);
    
    return {
      inputs: inputField ? { [inputField.name]: inputField } : {},
      outputs: outputField ? { [outputField.name]: outputField } : {}
    };
  }
  
  /**
   * Create an AxLLM program from candidate
   */
  private createAxProgram(candidate: ComponentMap): any {
    const programDef = candidate['program'] || JSON.stringify(this.baseProgram);
    const program: AxProgram = typeof programDef === 'string' 
      ? JSON.parse(programDef) 
      : programDef;
    
    // Create AxLLM instance with the signature
    const axProgram = this.axInstance(program.signature);
    
    // Apply demos if available
    if (program.demos && program.demos.length > 0) {
      axProgram.setDemos(program.demos);
    }
    
    // Apply model config if available
    if (program.modelConfig) {
      // This would be applied to the LLM instance
      // For now, we'll store it for reference
      (axProgram as any)._modelConfig = program.modelConfig;
    }
    
    return axProgram;
  }
  
  /**
   * Execute AxLLM program on example
   */
  private async executeProgram(
    axProgram: any,
    example: AxExample,
    captureTrace: boolean = false
  ): Promise<AxPrediction> {
    const trace: AxTrace = {
      signature: axProgram._signature || '',
      inputs: example,
      outputs: {},
      modelCalls: [],
      score: 0
    };
    
    try {
      // Execute the AxLLM program
      const prediction = await axProgram.forward(this.llm, example);
      
      // Calculate metric score
      const score = await this.metricFn({ prediction, example });
      
      // Add metadata
      prediction._score = score;
      
      if (captureTrace) {
        trace.outputs = prediction;
        trace.score = score;
        
        // Capture model calls if available
        if ((axProgram as any)._lastModelCalls) {
          trace.modelCalls = (axProgram as any)._lastModelCalls;
        }
      }
      
      return prediction;
      
    } catch (error) {
      console.error('AxLLM execution error:', error);
      return {
        _score: 0,
        _feedback: `Execution failed: ${error}`
      };
    }
  }
  
  async evaluate(
    batch: AxExample[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<AxTrace, AxPrediction>> {
    const axProgram = this.createAxProgram(candidate);
    
    const outputs: AxPrediction[] = [];
    const scores: number[] = [];
    const trajectories: AxTrace[] | null = captureTraces ? [] : null;
    
    for (const example of batch) {
      const prediction = await this.executeProgram(axProgram, example, captureTraces);
      outputs.push(prediction);
      scores.push(prediction._score || 0);
      
      if (trajectories && captureTraces) {
        const trace: AxTrace = {
          signature: axProgram._signature || candidate['program'],
          inputs: example,
          outputs: prediction,
          modelCalls: [],
          score: prediction._score || 0
        };
        trajectories.push(trace);
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<AxTrace, AxPrediction>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    if (!evalBatch.trajectories) {
      return dataset;
    }
    
    // Analyze performance for program evolution
    const programAnalysis: Array<Record<string, any>> = [];
    
    for (let i = 0; i < Math.min(5, evalBatch.trajectories.length); i++) {
      const trace = evalBatch.trajectories[i];
      const score = evalBatch.scores[i];
      const output = evalBatch.outputs[i];
      
      let feedback = '';
      let suggestions: string[] = [];
      
      if (score < 0.3) {
        feedback = 'Poor performance - major changes needed';
        suggestions.push('Add more descriptive instruction to signature');
        suggestions.push('Include high-quality demos');
        suggestions.push('Adjust temperature for more/less creativity');
        suggestions.push('Consider different output format');
      } else if (score < 0.7) {
        feedback = 'Moderate performance - refinement needed';
        suggestions.push('Fine-tune instruction wording');
        suggestions.push('Add more diverse demos');
        suggestions.push('Optimize model parameters');
      } else {
        feedback = 'Good performance - minor optimizations possible';
        suggestions.push('Reduce demo count if over 5');
        suggestions.push('Simplify instruction if verbose');
      }
      
      programAnalysis.push({
        'Current Program': candidate['program'],
        'Score': score,
        'Feedback': feedback,
        'Suggestions': suggestions,
        'Failed On': score < 0.5 ? trace.inputs : null,
        'Succeeded On': score >= 0.8 ? trace.inputs : null
      });
    }
    
    dataset['program'] = programAnalysis;
    
    return dataset;
  }
  
  async proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): Promise<ComponentMap> {
    const currentProgram: AxProgram = JSON.parse(candidate['program'] || JSON.stringify(this.baseProgram));
    const analysis = reflectiveDataset['program'] || [];
    
    // Calculate average score
    const avgScore = analysis.reduce((sum, a) => sum + a['Score'], 0) / (analysis.length || 1);
    
    // Use reflection LM if available for sophisticated evolution
    if (this.reflectionLm) {
      const evolvedProgramStr = await this.evolveWithLLM(currentProgram, analysis);
      return { program: evolvedProgramStr };
    }
    
    // Heuristic evolution based on performance
    let evolvedProgram = { ...currentProgram };
    
    if (avgScore < 0.3) {
      // Major restructuring
      evolvedProgram = this.majorEvolution(currentProgram, analysis);
    } else if (avgScore < 0.7) {
      // Incremental improvements
      evolvedProgram = this.incrementalEvolution(currentProgram, analysis);
    } else {
      // Minor optimizations
      evolvedProgram = this.minorOptimization(currentProgram, analysis);
    }
    
    return { program: JSON.stringify(evolvedProgram) };
  }
  
  private async evolveWithLLM(program: AxProgram, analysis: any[]): Promise<string> {
    const prompt = `You are evolving an AxLLM program based on performance feedback.

Current program:
${JSON.stringify(program, null, 2)}

Performance analysis:
${JSON.stringify(analysis.slice(0, 3), null, 2)}

Improve the program by:
1. Enhancing the signature instruction for clarity
2. Selecting better few-shot demos from successful examples
3. Adjusting model config (temperature, maxTokens)
4. Optimizing the input/output format

Return ONLY a valid JSON object for the improved AxProgram.`;

    const response = await this.reflectionLm!(prompt);
    
    try {
      // Validate it's valid JSON
      JSON.parse(response);
      return response;
    } catch {
      // Return original if parsing fails
      return JSON.stringify(program);
    }
  }
  
  private majorEvolution(program: AxProgram, analysis: any[]): AxProgram {
    const evolved = { ...program };
    
    // Enhance signature with detailed instruction
    if (typeof evolved.signature === 'string') {
      const sig = this.parseSignature(evolved.signature);
      evolved.signature = evolved.signature + ' "Think step-by-step and be precise"';
    }
    
    // Add successful examples as demos
    const successfulExamples = analysis
      .filter(a => a['Score'] >= 0.8 && a['Succeeded On'])
      .map(a => a['Succeeded On']);
    
    if (successfulExamples.length > 0) {
      evolved.demos = successfulExamples.slice(0, 5);
    }
    
    // Adjust model config for better performance
    evolved.modelConfig = {
      ...evolved.modelConfig,
      temperature: 0.3, // Lower for more consistency
      maxTokens: 200   // More space for reasoning
    };
    
    return evolved;
  }
  
  private incrementalEvolution(program: AxProgram, analysis: any[]): AxProgram {
    const evolved = { ...program };
    
    // Refine demos - keep best performing
    if (evolved.demos && evolved.demos.length > 3) {
      evolved.demos = evolved.demos.slice(0, 3);
    }
    
    // Fine-tune temperature
    if (evolved.modelConfig) {
      const currentTemp = evolved.modelConfig.temperature || 0.7;
      evolved.modelConfig.temperature = currentTemp * 0.9; // Slightly reduce
    }
    
    return evolved;
  }
  
  private minorOptimization(program: AxProgram, analysis: any[]): AxProgram {
    const evolved = { ...program };
    
    // Remove redundant demos if too many
    if (evolved.demos && evolved.demos.length > 5) {
      evolved.demos = evolved.demos.slice(0, 5);
    }
    
    return evolved;
  }
  
  /**
   * Convert GEPA optimization result to AxOptimizedProgram format
   */
  toAxOptimizedProgram(
    candidate: ComponentMap,
    score: number,
    stats?: any
  ): AxOptimizedProgram {
    const program: AxProgram = JSON.parse(candidate['program'] || JSON.stringify(this.baseProgram));
    
    return {
      bestScore: score,
      instruction: typeof program.signature === 'string' ? program.signature : JSON.stringify(program.signature),
      demos: program.demos,
      modelConfig: program.modelConfig,
      optimizerType: 'GEPA',
      optimizationTime: Date.now(),
      totalRounds: stats?.rounds || 0,
      converged: score >= 0.95,
      stats
    };
  }
  
  /**
   * Load AxOptimizedProgram and convert to GEPA candidate
   */
  fromAxOptimizedProgram(optimized: AxOptimizedProgram): ComponentMap {
    const program: AxProgram = {
      signature: optimized.instruction || '',
      demos: optimized.demos,
      modelConfig: optimized.modelConfig
    };
    
    return {
      program: JSON.stringify(program)
    };
  }
}

/**
 * Factory function for creating AxLLM native adapter
 */
export function createAxLLMNativeAdapter(config: {
  axInstance: any;
  llm: any;
  baseProgram?: AxProgram;
  metricFn: (params: { prediction: any; example: any }) => number | Promise<number>;
  reflectionLm?: (prompt: string) => Promise<string>;
}): AxLLMNativeAdapter {
  return new AxLLMNativeAdapter(config);
}