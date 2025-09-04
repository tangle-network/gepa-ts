import { 
  BaseAdapter, 
  EvaluationBatch 
} from '../core/adapter.js';
import { 
  ComponentMap, 
  ReflectiveDataset, 
  DataInst, 
  Trajectory, 
  RolloutOutput 
} from '../types/index.js';

// DSPy types
export interface DSPySignature {
  instructions: string;
  inputFields: Record<string, FieldDescriptor>;
  outputFields: Record<string, FieldDescriptor>;
}

export interface FieldDescriptor {
  name: string;
  description: string;
  type?: string;
  required?: boolean;
}

export interface DSPyModule {
  name: string;
  signature: DSPySignature;
  type: 'Predict' | 'ChainOfThought' | 'ReAct' | 'Custom';
  submodules?: DSPyModule[];
}

export interface DSPyProgram {
  modules: DSPyModule[];
  flow: string; // Description of control flow
}

export interface DSPyExample extends DataInst {
  inputs: Record<string, any>;
  outputs?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface DSPyTrace extends Trajectory {
  moduleInvocations: Array<{
    moduleName: string;
    inputs: Record<string, any>;
    outputs: Record<string, any>;
    reasoning?: string;
    timestamp: number;
  }>;
  score: number;
  feedback?: string;
  errors?: string[];
}

export interface DSPyPrediction extends RolloutOutput {
  outputs: Record<string, any>;
  reasoning?: string;
  score: number;
  trace?: DSPyTrace;
}

export interface DSPyAdapterConfig {
  studentModule: DSPyProgram;
  metricFn: (example: DSPyExample, prediction: DSPyPrediction) => number | Promise<number>;
  feedbackMap?: Record<string, (
    predictorOutput: Record<string, any>,
    predictorInputs: Record<string, any>,
    moduleInputs: DSPyExample,
    moduleOutputs: DSPyPrediction,
    trace: DSPyTrace
  ) => { score: number; feedback: string }>;
  failureScore?: number;
  numThreads?: number;
  addFormatFailureAsFeedback?: boolean;
  lm?: (prompt: string) => Promise<string> | string;
}

export class DSPyAdapter extends BaseAdapter<DSPyExample, DSPyTrace, DSPyPrediction> {
  private config: DSPyAdapterConfig;
  private namedPredictors: Map<string, DSPyModule>;
  
  constructor(config: DSPyAdapterConfig) {
    super();
    this.config = {
      failureScore: 0,
      addFormatFailureAsFeedback: false,
      ...config
    };
    
    // Cache named predictors
    this.namedPredictors = new Map();
    this.extractNamedPredictors(config.studentModule);
  }
  
  private extractNamedPredictors(program: DSPyProgram): void {
    const extract = (modules: DSPyModule[], prefix: string = '') => {
      for (const module of modules) {
        const name = prefix ? `${prefix}.${module.name}` : module.name;
        this.namedPredictors.set(name, module);
        
        if (module.submodules) {
          extract(module.submodules, name);
        }
      }
    };
    extract(program.modules);
  }
  
  private buildProgram(candidate: ComponentMap): DSPyProgram {
    // Deep copy the student module
    const newProgram: DSPyProgram = JSON.parse(JSON.stringify(this.config.studentModule));
    
    // Update instructions for named predictors
    const updateModules = (modules: DSPyModule[], prefix: string = '') => {
      for (const module of modules) {
        const name = prefix ? `${prefix}.${module.name}` : module.name;
        
        if (candidate[name]) {
          module.signature.instructions = candidate[name];
        }
        
        if (module.submodules) {
          updateModules(module.submodules, name);
        }
      }
    };
    
    updateModules(newProgram.modules);
    return newProgram;
  }
  
  private async executeModule(
    module: DSPyModule,
    inputs: Record<string, any>,
    captureTrace: boolean = false
  ): Promise<{ outputs: Record<string, any>; trace?: any }> {
    const prompt = this.buildPrompt(module, inputs);
    
    if (!this.config.lm) {
      // Mock execution for testing
      return {
        outputs: Object.keys(module.signature.outputFields).reduce((acc, field) => {
          acc[field] = `Generated ${field}`;
          return acc;
        }, {} as Record<string, any>),
        trace: captureTrace ? { prompt, response: 'Mock response' } : undefined
      };
    }
    
    const response = await this.config.lm(prompt);
    const outputs = this.parseResponse(response, module.signature.outputFields);
    
    return {
      outputs,
      trace: captureTrace ? { prompt, response } : undefined
    };
  }
  
  private buildPrompt(module: DSPyModule, inputs: Record<string, any>): string {
    const parts: string[] = [];
    
    // Add instructions
    parts.push(module.signature.instructions);
    parts.push('');
    
    // Add input fields
    for (const [fieldName, descriptor] of Object.entries(module.signature.inputFields)) {
      const value = inputs[fieldName];
      parts.push(`${descriptor.description}:`);
      parts.push(`${fieldName}: ${JSON.stringify(value)}`);
      parts.push('');
    }
    
    // Add output field descriptions
    parts.push('Please provide the following outputs:');
    for (const [fieldName, descriptor] of Object.entries(module.signature.outputFields)) {
      parts.push(`${fieldName}: ${descriptor.description}`);
    }
    
    // Add Chain of Thought if needed
    if (module.type === 'ChainOfThought') {
      parts.unshift('Think step-by-step to solve this problem.');
      parts.push('');
      parts.push('First, provide your reasoning, then give the final outputs.');
    }
    
    return parts.join('\n');
  }
  
  private parseResponse(
    response: string,
    outputFields: Record<string, FieldDescriptor>
  ): Record<string, any> {
    const outputs: Record<string, any> = {};
    
    // Simple parsing - look for field names in response
    for (const fieldName of Object.keys(outputFields)) {
      const regex = new RegExp(`${fieldName}:\\s*(.+?)(?:\\n|$)`, 'i');
      const match = response.match(regex);
      
      if (match) {
        outputs[fieldName] = match[1].trim();
      } else {
        outputs[fieldName] = '';
      }
    }
    
    // Extract reasoning for ChainOfThought
    const reasoningMatch = response.match(/reasoning:?\s*([\s\S]+?)(?:answer|output|$)/i);
    if (reasoningMatch) {
      outputs.reasoning = reasoningMatch[1].trim();
    }
    
    return outputs;
  }
  
  private async executeProgram(
    program: DSPyProgram,
    example: DSPyExample,
    captureTraces: boolean = false
  ): Promise<DSPyPrediction> {
    const trace: DSPyTrace = {
      moduleInvocations: [],
      score: 0
    };
    
    let currentInputs = example.inputs;
    let finalOutputs: Record<string, any> = {};
    
    // Execute modules in sequence (simplified flow)
    for (const module of program.modules) {
      const startTime = Date.now();
      
      try {
        const { outputs, trace: moduleTrace } = await this.executeModule(
          module,
          currentInputs,
          captureTraces
        );
        
        if (captureTraces) {
          trace.moduleInvocations.push({
            moduleName: module.name,
            inputs: { ...currentInputs },
            outputs: { ...outputs },
            reasoning: outputs.reasoning,
            timestamp: startTime
          });
        }
        
        // Update inputs for next module
        currentInputs = { ...currentInputs, ...outputs };
        finalOutputs = { ...finalOutputs, ...outputs };
        
      } catch (error) {
        if (captureTraces) {
          trace.errors = trace.errors || [];
          trace.errors.push(`Error in ${module.name}: ${error}`);
        }
        
        // Return failure
        return {
          outputs: finalOutputs,
          score: this.config.failureScore || 0,
          trace: captureTraces ? trace : undefined
        };
      }
    }
    
    // Calculate score
    const score = await this.config.metricFn(example, {
      outputs: finalOutputs,
      score: 0,
      trace
    });
    
    return {
      outputs: finalOutputs,
      score,
      trace: captureTraces ? { ...trace, score } : undefined
    };
  }
  
  async evaluate(
    batch: DSPyExample[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<DSPyTrace, DSPyPrediction>> {
    const program = this.buildProgram(candidate);
    const outputs: DSPyPrediction[] = [];
    const scores: number[] = [];
    const trajectories: DSPyTrace[] | null = captureTraces ? [] : null;
    
    // Execute in parallel if possible
    const promises = batch.map(example => 
      this.executeProgram(program, example, captureTraces)
    );
    
    const results = await Promise.all(promises);
    
    for (const prediction of results) {
      outputs.push(prediction);
      scores.push(prediction.score);
      
      if (trajectories && prediction.trace) {
        trajectories.push(prediction.trace);
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<DSPyTrace, DSPyPrediction>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    if (!evalBatch.trajectories) {
      return dataset;
    }
    
    for (const componentName of componentsToUpdate) {
      const module = this.namedPredictors.get(componentName);
      if (!module) continue;
      
      const items: Array<Record<string, any>> = [];
      
      for (let i = 0; i < evalBatch.trajectories.length; i++) {
        const trace = evalBatch.trajectories[i];
        const prediction = evalBatch.outputs[i];
        const score = evalBatch.scores[i];
        
        // Find invocations of this module
        const moduleInvocations = trace.moduleInvocations.filter(
          inv => inv.moduleName === componentName
        );
        
        for (const invocation of moduleInvocations) {
          let feedback = '';
          
          // Use custom feedback if available
          if (this.config.feedbackMap && this.config.feedbackMap[componentName]) {
            const result = this.config.feedbackMap[componentName](
              invocation.outputs,
              invocation.inputs,
              { inputs: invocation.inputs } as DSPyExample,
              prediction,
              trace
            );
            feedback = result.feedback;
          } else {
            // Default feedback
            if (score < 0.5) {
              feedback = 'Output did not meet quality standards. ';
              if (trace.errors?.length) {
                feedback += `Errors: ${trace.errors.join(', ')}`;
              }
            } else if (score < 0.8) {
              feedback = 'Output partially correct but could be improved.';
            } else {
              feedback = 'Good output.';
            }
          }
          
          items.push({
            'Module': componentName,
            'Inputs': invocation.inputs,
            'Generated Outputs': invocation.outputs,
            'Score': score,
            'Feedback': feedback,
            'Current Instructions': module.signature.instructions
          });
        }
      }
      
      // Limit to most informative examples
      dataset[componentName] = items.slice(0, 5);
    }
    
    return dataset;
  }
  
  // Support for full program evolution
  proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): ComponentMap {
    const newTexts: ComponentMap = {};
    
    for (const componentName of componentsToUpdate) {
      const examples = reflectiveDataset[componentName] || [];
      
      // Analyze feedback to generate improved instructions
      const failurePatterns: string[] = [];
      const successPatterns: string[] = [];
      
      for (const example of examples) {
        if (example['Score'] < 0.5) {
          failurePatterns.push(example['Feedback']);
        } else if (example['Score'] > 0.8) {
          successPatterns.push(JSON.stringify(example['Generated Outputs']));
        }
      }
      
      // Generate improved instruction
      let improvedInstruction = candidate[componentName] || '';
      
      if (failurePatterns.length > 0) {
        improvedInstruction += '\n\nCommon issues to avoid:\n';
        improvedInstruction += failurePatterns.slice(0, 3).map((p, i) => `${i + 1}. ${p}`).join('\n');
      }
      
      if (successPatterns.length > 0) {
        improvedInstruction += '\n\nSuccessful output patterns:\n';
        improvedInstruction += '- Ensure outputs are structured and complete\n';
        improvedInstruction += '- Provide clear reasoning when applicable';
      }
      
      newTexts[componentName] = improvedInstruction;
    }
    
    return newTexts;
  }
}