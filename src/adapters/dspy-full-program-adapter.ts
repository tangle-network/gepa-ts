import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, ReflectiveDataset } from '../types/index.js';
import { DSPyExample, DSPyTrace, DSPyPrediction, DSPyProgram, DSPyModule } from './dspy-adapter.js';

/**
 * DSPy Full Program Adapter - Evolves entire DSPy programs including
 * signatures, modules, and control flow logic.
 * 
 * This adapter goes beyond just optimizing instructions - it can:
 * - Add/remove modules
 * - Change module types (Predict -> ChainOfThought)
 * - Modify control flow
 * - Evolve custom module implementations
 */
export class DSPyFullProgramAdapter extends BaseAdapter<DSPyExample, DSPyTrace, DSPyPrediction> {
  private metricFn: (example: DSPyExample, prediction: DSPyPrediction) => number | Promise<number>;
  private lm?: (prompt: string) => Promise<string> | string;
  private baseProgram: string; // String representation of the program
  
  constructor(config: {
    baseProgram: string;
    metricFn: (example: DSPyExample, prediction: DSPyPrediction) => number | Promise<number>;
    lm?: (prompt: string) => Promise<string> | string;
  }) {
    super();
    this.baseProgram = config.baseProgram;
    this.metricFn = config.metricFn;
    this.lm = config.lm;
  }
  
  private parseProgram(programCode: string): DSPyProgram {
    // Parse the program string into a DSPyProgram structure
    // This is simplified - in reality would use proper AST parsing
    
    const modules: DSPyModule[] = [];
    
    // Extract module definitions
    const moduleRegex = /dspy\.(Predict|ChainOfThought|ReAct)\(["']([^"']+)["']\)/g;
    let match;
    let moduleIndex = 0;
    
    while ((match = moduleRegex.exec(programCode)) !== null) {
      const [, type, signature] = match;
      modules.push({
        name: `module_${moduleIndex++}`,
        type: type as any,
        signature: this.parseSignature(signature)
      });
    }
    
    // Extract custom module classes
    const classRegex = /class (\w+)\(dspy\.Module\):([\s\S]*?)(?=class|\Z)/g;
    while ((match = classRegex.exec(programCode)) !== null) {
      const [, className, classBody] = match;
      const submodules = this.extractSubmodules(classBody);
      
      modules.push({
        name: className,
        type: 'Custom',
        signature: {
          instructions: `Custom module ${className}`,
          inputFields: {},
          outputFields: {}
        },
        submodules
      });
    }
    
    return {
      modules,
      flow: programCode
    };
  }
  
  private parseSignature(sig: string): any {
    // Parse signature string like "question -> answer"
    const parts = sig.split('->').map(s => s.trim());
    
    return {
      instructions: `Process ${parts[0]} to produce ${parts[1]}`,
      inputFields: {
        [parts[0]]: { name: parts[0], description: `Input ${parts[0]}` }
      },
      outputFields: {
        [parts[1]]: { name: parts[1], description: `Output ${parts[1]}` }
      }
    };
  }
  
  private extractSubmodules(classBody: string): DSPyModule[] {
    const submodules: DSPyModule[] = [];
    const initRegex = /self\.(\w+)\s*=\s*dspy\.(Predict|ChainOfThought|ReAct)\(["']([^"']+)["']\)/g;
    let match;
    
    while ((match = initRegex.exec(classBody)) !== null) {
      const [, name, type, signature] = match;
      submodules.push({
        name,
        type: type as any,
        signature: this.parseSignature(signature)
      });
    }
    
    return submodules;
  }
  
  private async executeProgram(
    programCode: string,
    example: DSPyExample,
    captureTrace: boolean = false
  ): Promise<DSPyPrediction> {
    // In a real implementation, this would execute the Python code
    // For now, we'll simulate execution
    
    const trace: DSPyTrace = {
      moduleInvocations: [],
      score: 0
    };
    
    try {
      // Simulate program execution
      const outputs: Record<string, any> = {};
      
      // Extract expected output structure from program
      if (programCode.includes('ChainOfThought')) {
        outputs.reasoning = 'Step-by-step reasoning...';
        outputs.answer = 'Generated answer';
      } else if (programCode.includes('ReAct')) {
        outputs.thought = 'Initial thought';
        outputs.action = 'Take action';
        outputs.observation = 'Observe result';
        outputs.answer = 'Final answer';
      } else {
        outputs.answer = 'Direct answer';
      }
      
      // Calculate score
      const prediction: DSPyPrediction = {
        outputs,
        score: 0,
        trace: captureTrace ? trace : undefined
      };
      
      prediction.score = await this.metricFn(example, prediction);
      
      if (captureTrace) {
        trace.score = prediction.score;
        trace.moduleInvocations.push({
          moduleName: 'program',
          inputs: example.inputs,
          outputs,
          timestamp: Date.now()
        });
      }
      
      return prediction;
      
    } catch (error) {
      return {
        outputs: {},
        score: 0,
        trace: captureTrace ? { ...trace, errors: [`Execution error: ${error}`] } : undefined
      };
    }
  }
  
  async evaluate(
    batch: DSPyExample[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<DSPyTrace, DSPyPrediction>> {
    const programCode = candidate['program'] || this.baseProgram;
    
    const outputs: DSPyPrediction[] = [];
    const scores: number[] = [];
    const trajectories: DSPyTrace[] | null = captureTraces ? [] : null;
    
    for (const example of batch) {
      const prediction = await this.executeProgram(programCode, example, captureTraces);
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
    
    // For full program evolution, we analyze the entire program
    const programAnalysis: Array<Record<string, any>> = [];
    
    for (let i = 0; i < evalBatch.trajectories.length; i++) {
      const trace = evalBatch.trajectories[i];
      const score = evalBatch.scores[i];
      const output = evalBatch.outputs[i];
      
      let feedback = '';
      let suggestions = [];
      
      if (score < 0.3) {
        feedback = 'Program failed to produce correct output.';
        suggestions.push('Consider adding ChainOfThought for complex reasoning');
        suggestions.push('Add intermediate validation steps');
        suggestions.push('Improve error handling');
      } else if (score < 0.7) {
        feedback = 'Program partially successful.';
        suggestions.push('Refine module signatures for clarity');
        suggestions.push('Add more specific instructions');
        
        // Analyze specific failure modes
        if (!output.outputs.reasoning && trace.moduleInvocations.length > 1) {
          suggestions.push('Add reasoning trace for debugging');
        }
      } else {
        feedback = 'Program performing well.';
        suggestions.push('Consider optimizing for efficiency');
        suggestions.push('Simplify if over-engineered');
      }
      
      programAnalysis.push({
        'Current Program': candidate['program'] || this.baseProgram,
        'Score': score,
        'Feedback': feedback,
        'Suggestions': suggestions,
        'Failed Examples': score < 0.5 ? trace.moduleInvocations : [],
        'Successful Patterns': score > 0.8 ? output.outputs : {}
      });
    }
    
    dataset['program'] = programAnalysis.slice(0, 5);
    
    return dataset;
  }
  
  proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): ComponentMap {
    const currentProgram = candidate['program'] || this.baseProgram;
    const examples = reflectiveDataset['program'] || [];
    
    // Analyze feedback to determine program evolution strategy
    const avgScore = examples.reduce((sum, ex) => sum + ex['Score'], 0) / (examples.length || 1);
    
    let evolvedProgram = currentProgram;
    
    if (avgScore < 0.3) {
      // Major restructuring needed
      evolvedProgram = this.majorRestructure(currentProgram, examples);
    } else if (avgScore < 0.7) {
      // Incremental improvements
      evolvedProgram = this.incrementalImprove(currentProgram, examples);
    } else {
      // Minor optimizations
      evolvedProgram = this.optimize(currentProgram, examples);
    }
    
    return { program: evolvedProgram };
  }
  
  private majorRestructure(program: string, feedback: any[]): string {
    // Convert simple Predict to ChainOfThought
    let evolved = program.replace(/dspy\.Predict\(/g, 'dspy.ChainOfThought(');
    
    // Add error handling
    if (!evolved.includes('try:')) {
      evolved = evolved.replace(
        /def forward\(self,/,
        'def forward(self,\n        try:\n            '
      );
    }
    
    // Add intermediate steps
    const suggestions = feedback.flatMap(f => f['Suggestions'] || []);
    if (suggestions.some(s => s.includes('intermediate'))) {
      evolved = evolved.replace(
        'return dspy.Prediction(',
        '# Add intermediate validation\n        ' +
        'intermediate = self.validate(result)\n        ' +
        'return dspy.Prediction('
      );
    }
    
    return evolved;
  }
  
  private incrementalImprove(program: string, feedback: any[]): string {
    let evolved = program;
    
    // Improve signatures
    const signatureRegex = /"([^"]+)"/g;
    evolved = evolved.replace(signatureRegex, (match, sig) => {
      const parts = sig.split('->').map(s => s.trim());
      if (parts.length === 2) {
        return `"${parts[0]} -> reasoning, ${parts[1]}"`;
      }
      return match;
    });
    
    // Add detailed docstrings
    if (!evolved.includes('"""')) {
      const classMatch = evolved.match(/class (\w+)\(dspy\.Module\):/);
      if (classMatch) {
        evolved = evolved.replace(
          classMatch[0],
          `${classMatch[0]}\n    """Enhanced module with improved error handling and reasoning."""`
        );
      }
    }
    
    return evolved;
  }
  
  private optimize(program: string, feedback: any[]): string {
    // Remove redundant modules
    let evolved = program;
    
    // Simplify if over-engineered
    const moduleCount = (evolved.match(/dspy\.\w+\(/g) || []).length;
    if (moduleCount > 5) {
      // Consider consolidating modules
      evolved = `# Consider consolidating modules for efficiency\n${evolved}`;
    }
    
    return evolved;
  }
}