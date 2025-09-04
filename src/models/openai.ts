/**
 * OpenAI Language Model integration for GEPA
 */

export interface OpenAIConfig {
  apiKey: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  baseURL?: string;
}

export class OpenAILanguageModel {
  private config: Required<Omit<OpenAIConfig, 'baseURL'>> & { baseURL?: string };
  
  constructor(config: OpenAIConfig) {
    this.config = {
      apiKey: config.apiKey,
      model: config.model || 'gpt-3.5-turbo',
      temperature: config.temperature ?? 0.7,
      maxTokens: config.maxTokens ?? 1000,
      topP: config.topP ?? 1,
      frequencyPenalty: config.frequencyPenalty ?? 0,
      presencePenalty: config.presencePenalty ?? 0,
      baseURL: config.baseURL
    };
  }
  
  /**
   * Generate text using OpenAI API
   */
  async generate(prompt: string): Promise<string> {
    const url = this.config.baseURL 
      ? `${this.config.baseURL}/chat/completions`
      : 'https://api.openai.com/v1/chat/completions';
    
    const requestBody = {
      model: this.config.model,
      messages: [
        { role: 'user', content: prompt }
      ],
      temperature: this.config.temperature,
      max_tokens: this.config.maxTokens,
      top_p: this.config.topP,
      frequency_penalty: this.config.frequencyPenalty,
      presence_penalty: this.config.presencePenalty
    };
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
      }
      
      const data = await response.json();
      
      if (!data.choices || data.choices.length === 0) {
        throw new Error('No choices returned from OpenAI API');
      }
      
      return data.choices[0].message.content || '';
      
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`OpenAI API request failed: ${error}`);
    }
  }
  
  /**
   * Generate multiple completions
   */
  async generateBatch(prompts: string[]): Promise<string[]> {
    const results = await Promise.all(
      prompts.map(prompt => this.generate(prompt))
    );
    return results;
  }
  
  /**
   * Stream generation (for future use)
   */
  async *generateStream(prompt: string): AsyncGenerator<string, void, unknown> {
    // For now, just yield the full response
    // In a full implementation, this would use the streaming API
    const response = await this.generate(prompt);
    yield response;
  }
  
  /**
   * Get model configuration
   */
  getConfig(): Readonly<OpenAIConfig> {
    return { ...this.config };
  }
  
  /**
   * Update configuration
   */
  updateConfig(updates: Partial<OpenAIConfig>): void {
    this.config = { ...this.config, ...updates };
  }
}

/**
 * Factory function for creating OpenAI models
 */
export function createOpenAIModel(config: OpenAIConfig): OpenAILanguageModel {
  return new OpenAILanguageModel(config);
}

/**
 * Simple function interface for LLM calls
 */
export function createOpenAIFunction(config: OpenAIConfig): (prompt: string) => Promise<string> {
  const model = new OpenAILanguageModel(config);
  return (prompt: string) => model.generate(prompt);
}