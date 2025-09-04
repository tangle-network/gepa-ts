import { LanguageModel } from '../types/index.js';

const INSTRUCTION_PROPOSAL_PROMPT = `
You are an expert in optimizing instructions for language models.

You will be given:
1. The current instruction being used
2. A dataset showing examples of inputs, outputs, and feedback

Your task is to analyze the feedback and propose an improved instruction that addresses the issues identified.

Current Instruction:
{currentInstruction}

Dataset with Feedback:
{datasetWithFeedback}

Based on the feedback, propose an improved instruction that:
1. Addresses the specific errors or issues identified
2. Maintains clarity and conciseness
3. Preserves what's working well
4. Adds specific guidance where needed

Return ONLY the new instruction text, without any explanation or markup.
`;

export async function proposeInstructionWithLM(
  lm: LanguageModel,
  currentInstruction: string,
  datasetWithFeedback: Array<Record<string, any>>
): Promise<string> {
  const prompt = INSTRUCTION_PROPOSAL_PROMPT
    .replace('{currentInstruction}', currentInstruction)
    .replace('{datasetWithFeedback}', JSON.stringify(datasetWithFeedback, null, 2));
  
  const response = lm(prompt);
  const result = response instanceof Promise ? await response : response;
  
  return result.trim();
}

export class InstructionProposer {
  private lm: LanguageModel;
  
  constructor(lm: LanguageModel) {
    this.lm = lm;
  }
  
  async propose(
    currentInstructions: Record<string, string>,
    reflectiveDatasets: Record<string, Array<Record<string, any>>>,
    componentsToUpdate: string[]
  ): Promise<Record<string, string>> {
    const newInstructions: Record<string, string> = {};
    
    for (const componentName of componentsToUpdate) {
      if (!(componentName in currentInstructions)) {
        throw new Error(`Component ${componentName} not found in current instructions`);
      }
      
      const currentInstruction = currentInstructions[componentName];
      const dataset = reflectiveDatasets[componentName] || [];
      
      newInstructions[componentName] = await proposeInstructionWithLM(
        this.lm,
        currentInstruction,
        dataset
      );
    }
    
    return newInstructions;
  }
}