import { 
  ComponentMap, 
  DataInst, 
  ReflectiveDataset, 
  RolloutOutput, 
  Trajectory 
} from '../types/index.js';
import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';

export interface DefaultDataInst {
  question: string;
  answer?: string;
  [key: string]: any;
}

export interface DefaultTrajectory {
  input: DefaultDataInst;
  output: string;
  expected?: string;
  correct?: boolean;
}

export class DefaultAdapter extends BaseAdapter<DefaultDataInst, DefaultTrajectory, string> {
  private model: string | ((prompt: string) => Promise<string> | string);
  
  constructor(model: string | ((prompt: string) => Promise<string> | string)) {
    super();
    this.model = model;
  }
  
  private async callModel(prompt: string): Promise<string> {
    if (typeof this.model === 'function') {
      const result = this.model(prompt);
      return result instanceof Promise ? await result : result;
    }
    
    const { OpenAI } = await import('openai');
    const openai = new OpenAI();
    
    const response = await openai.chat.completions.create({
      model: this.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0
    });
    
    return response.choices[0].message.content || '';
  }
  
  async evaluate(
    batch: DefaultDataInst[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<DefaultTrajectory, string>> {
    const systemPrompt = candidate['system_prompt'] || '';
    const outputs: string[] = [];
    const scores: number[] = [];
    const trajectories: DefaultTrajectory[] | null = captureTraces ? [] : null;
    
    for (const dataInst of batch) {
      const prompt = `${systemPrompt}\n\n${dataInst.question}`;
      
      try {
        const output = await this.callModel(prompt);
        outputs.push(output);
        
        let score = 0;
        if (dataInst.answer) {
          const normalizedOutput = output.toLowerCase().trim();
          const normalizedAnswer = dataInst.answer.toLowerCase().trim();
          score = normalizedOutput.includes(normalizedAnswer) ? 1 : 0;
        } else {
          score = output.length > 0 ? 0.5 : 0;
        }
        
        scores.push(score);
        
        if (trajectories) {
          trajectories.push({
            input: dataInst,
            output,
            expected: dataInst.answer,
            correct: score === 1
          });
        }
      } catch (error) {
        outputs.push('');
        scores.push(0);
        
        if (trajectories) {
          trajectories.push({
            input: dataInst,
            output: `Error: ${error}`,
            expected: dataInst.answer,
            correct: false
          });
        }
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<DefaultTrajectory, string>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    if (!evalBatch.trajectories) {
      return dataset;
    }
    
    for (const componentName of componentsToUpdate) {
      if (componentName === 'system_prompt') {
        const examples = evalBatch.trajectories.map((traj, idx) => ({
          'Input': traj.input.question,
          'Generated Output': traj.output,
          'Expected Output': traj.expected || 'N/A',
          'Score': evalBatch.scores[idx],
          'Feedback': traj.correct ? 
            'Correct answer' : 
            `Incorrect. Expected: ${traj.expected}, Got: ${traj.output}`
        }));
        
        dataset[componentName] = examples.slice(0, 5);
      }
    }
    
    return dataset;
  }
}