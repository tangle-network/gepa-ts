import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, ReflectiveDataset } from '../types/index.js';
import { DSPyExample, DSPyTrace, DSPyPrediction, DSPyProgram, DSPyModule } from './dspy-adapter.js';
import { DSPyExecutor, createDSPyExecutor } from '../executors/dspy-executor.js';

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
  private taskLm?: (prompt: string) => Promise<string> | string;
  private reflectionLm?: (prompt: string) => Promise<string> | string;
  private baseProgram: string; // String representation of the program
  private executor: DSPyExecutor;
  private numThreads: number;
  private taskLmConfig: any;
  
  constructor(config: {
    baseProgram?: string;
    taskLm?: any; // DSPy LM config or function
    metricFn: (example: DSPyExample, prediction: DSPyPrediction) => any;
    reflectionLm?: (prompt: string) => Promise<string> | string;
    numThreads?: number;
    pythonPath?: string;
  }) {
    super();
    this.baseProgram = config.baseProgram || 'import dspy\nprogram = dspy.ChainOfThought("question -> answer")';
    this.metricFn = config.metricFn;
    this.reflectionLm = config.reflectionLm;
    this.numThreads = config.numThreads || 10;
    this.executor = createDSPyExecutor(config.pythonPath);
    
    // Parse task LM config
    if (typeof config.taskLm === 'function') {
      this.taskLm = config.taskLm;
      this.taskLmConfig = { model: 'openai/gpt-4o-mini' };
    } else if (config.taskLm) {
      this.taskLmConfig = {
        model: config.taskLm.model || 'openai/gpt-4o-mini',
        apiKey: config.taskLm.apiKey || process.env.OPENAI_API_KEY,
        maxTokens: config.taskLm.maxTokens || 2000,
        temperature: config.taskLm.temperature || 0.7
      };
    } else {
      this.taskLmConfig = { model: 'openai/gpt-4o-mini' };
    }
  }
  
  private async validateProgram(programCode: string): Promise<{ valid: boolean; error?: string }> {
    try {
      // Quick syntax validation
      if (!programCode.includes('import dspy')) {
        return { valid: false, error: 'Program must import dspy' };
      }
      if (!programCode.includes('program =')) {
        return { valid: false, error: 'Program must define a "program" variable' };
      }
      return { valid: true };
    } catch (error) {
      return { valid: false, error: String(error) };
    }
  }
  
  private convertMetricFnToCode(): string {
    // Convert the metric function to Python code
    // This is a simplified version - in production, would need more sophisticated conversion
    return `
def metric_fn(example, pred):
    # Default metric function
    if hasattr(example, 'answer'):
        pred_answer = pred.get('answer', '') if hasattr(pred, 'get') else getattr(pred, 'answer', '')
        return 1.0 if str(pred_answer).strip() == str(example.answer).strip() else 0.0
    return 0.0
`;
  }
  
  private async executeProgram(
    programCode: string,
    examples: DSPyExample[],
    captureTrace: boolean = false
  ): Promise<{ predictions: DSPyPrediction[], traces?: DSPyTrace[] }> {
    // Use real Python DSPy execution
    const metricCode = this.convertMetricFnToCode();
    
    try {
      const result = await this.executor.execute(
        programCode,
        examples,
        this.taskLmConfig,
        metricCode
      );
      
      if (!result.success) {
        throw new Error(result.error || 'Execution failed');
      }
      
      const predictions: DSPyPrediction[] = [];
      const traces: DSPyTrace[] = [];
      
      for (let i = 0; i < examples.length; i++) {
        const res = result.results![i];
        const prediction: DSPyPrediction = {
          outputs: res.outputs,
          score: res.score,
          feedback: res.feedback
        };
        predictions.push(prediction);
        
        if (captureTrace && result.traces) {
          const trace: DSPyTrace = {
            moduleInvocations: result.traces[i].map((t: any) => ({
              moduleName: t.module,
              inputs: t.inputs,
              outputs: t.outputs,
              timestamp: Date.now()
            })),
            score: res.score
          };
          traces.push(trace);
        }
      }
      
      return { predictions, traces: captureTrace ? traces : undefined };
      
    } catch (error) {
      // Return empty predictions on error
      const predictions = examples.map(() => ({
        outputs: {},
        score: 0,
        error: String(error)
      }));
      
      return { predictions };
    }
  }
  
  async evaluate(
    batch: DSPyExample[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<DSPyTrace, DSPyPrediction>> {
    const programCode = candidate['program'] || this.baseProgram;
    
    // Validate program before execution
    const validation = await this.validateProgram(programCode);
    if (!validation.valid) {
      // Return failure for all examples
      return {
        outputs: batch.map(() => ({ outputs: {}, score: 0, error: validation.error })),
        scores: batch.map(() => 0),
        trajectories: captureTraces ? batch.map(() => ({ moduleInvocations: [], score: 0, errors: [validation.error!] })) : null
      };
    }
    
    // Execute program on batch
    const { predictions, traces } = await this.executeProgram(programCode, batch, captureTraces);
    
    return {
      outputs: predictions,
      scores: predictions.map(p => p.score || 0),
      trajectories: traces || null
    };
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
  
  async proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): Promise<ComponentMap> {
    const currentProgram = candidate['program'] || this.baseProgram;
    const examples = reflectiveDataset['program'] || [];
    
    // If we have a reflection LM, use it for sophisticated evolution
    if (this.reflectionLm) {
      const prompt = this.buildEvolutionPrompt(currentProgram, examples);
      const evolvedProgram = await this.reflectionLm(prompt);
      
      // Extract code from response
      const codeMatch = evolvedProgram.match(/```python?\n([\s\S]*?)\n```/);
      if (codeMatch) {
        return { program: codeMatch[1] };
      }
      
      // Try to find program definition directly
      if (evolvedProgram.includes('import dspy') && evolvedProgram.includes('program =')) {
        return { program: evolvedProgram };
      }
    }
    
    // Fallback to heuristic evolution
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
  
  private buildEvolutionPrompt(currentProgram: string, feedback: any[]): string {
    return `You are evolving a DSPy program based on execution feedback.

Current program:
\`\`\`python
${currentProgram}
\`\`\`

Execution feedback:
${JSON.stringify(feedback.slice(0, 3), null, 2)}

Improve the program by:
1. Analyzing failure patterns
2. Adding error handling where needed
3. Improving module signatures
4. Optimizing the dataflow

Provide the improved program in a Python code block. Ensure it imports dspy and defines a 'program' variable.`;
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