/**
 * Unified Model Resolver
 *
 * Given a model ID string, resolves which provider adapter factory to use.
 * Resolution is two-stage:
 *   1. DB lookup  — accurate, reflects what's actually synced & configured.
 *   2. Heuristic  — prefix-based fallback for models not yet in the DB.
 *
 * Returns null when neither stage resolves → caller should return HTTP 400.
 */

import type { SupportedProvider } from "@shared";
import {
  anthropicAdapterFactory,
  azureAdapterFactory,
  bedrockAdapterFactory,
  cerebrasAdapterFactory,
  cohereAdapterFactory,
  deepseekAdapterFactory,
  geminiAdapterFactory,
  groqAdapterFactory,
  minimaxAdapterFactory,
  mistralAdapterFactory,
  ollamaAdapterFactory,
  openaiAdapterFactory,
  openrouterAdapterFactory,
  perplexityAdapterFactory,
  vllmAdapterFactory,
  xaiAdapterFactory,
  zhipuaiAdapterFactory,
} from "@/routes/proxy/adapters";
import logger from "@/logging";
import { ModelModel } from "@/models";
import type { LLMProvider } from "@/types";

// ---------------------------------------------------------------------------
// Adapter factory map  (one entry per SupportedProvider)
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyAdapterFactory = LLMProvider<any, any, any, any, any>;

const ADAPTER_FACTORIES: Partial<Record<SupportedProvider, AnyAdapterFactory>> =
  {
    openai: openaiAdapterFactory,
    anthropic: anthropicAdapterFactory,
    gemini: geminiAdapterFactory,
    bedrock: bedrockAdapterFactory,
    cohere: cohereAdapterFactory,
    cerebras: cerebrasAdapterFactory,
    mistral: mistralAdapterFactory,
    perplexity: perplexityAdapterFactory,
    groq: groqAdapterFactory,
    xai: xaiAdapterFactory,
    openrouter: openrouterAdapterFactory,
    vllm: vllmAdapterFactory,
    ollama: ollamaAdapterFactory,
    zhipuai: zhipuaiAdapterFactory,
    deepseek: deepseekAdapterFactory,
    minimax: minimaxAdapterFactory,
    azure: azureAdapterFactory,
  };

// ---------------------------------------------------------------------------
// Heuristic prefix table
// ---------------------------------------------------------------------------

/**
 * Maps well-known model-ID prefixes to a provider.
 * Entries are checked in declaration order — put more-specific prefixes first.
 */
const PREDETERMINED_PREFIXES: Array<[prefix: string, provider: SupportedProvider]> =
  [
    // OpenAI
    ["gpt-", "openai"],
    ["o1-", "openai"],
    ["o3-", "openai"],
    ["o4-", "openai"],
    ["text-embedding-", "openai"],
    ["dall-e-", "openai"],
    // Anthropic (direct)
    ["claude-", "anthropic"],
    // Bedrock (model IDs contain the provider prefix)
    ["anthropic.claude", "bedrock"],
    ["amazon.nova", "bedrock"],
    ["amazon.titan", "bedrock"],
    ["meta.llama", "bedrock"],
    ["mistral.mistral", "bedrock"],
    // Gemini
    ["gemini-", "gemini"],
    ["gemini/", "gemini"],
    // Cohere
    ["command-", "cohere"],
    ["embed-", "cohere"],
    // Mistral (direct, not Bedrock)
    ["mistral-", "mistral"],
    ["ministral-", "mistral"],
    ["codestral-", "mistral"],
    // Groq
    ["llama-", "groq"],
    ["mixtral-", "groq"],
    ["gemma2-", "groq"],
    // DeepSeek
    ["deepseek-", "deepseek"],
    // xAI
    ["grok-", "xai"],
    // Zhipu AI
    ["glm-", "zhipuai"],
    // MiniMax
    ["minimax-", "minimax"],
    ["MiniMax-", "minimax"],
    // Azure (Azure deployments often prefixed with "azure-")
    ["azure-", "azure"],
    // Perplexity
    ["sonar-", "perplexity"],
    ["sonar", "perplexity"],
    // Cerebras
    ["llama4-", "cerebras"],
  ];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface ResolvedModel {
  /** The provider that owns this model. */
  providerId: SupportedProvider;
  /** Ready-to-use adapter factory for handleLLMProxy(). */
  adapterFactory: AnyAdapterFactory;
  /** Whether the resolution came from the database or from prefix heuristics. */
  source: "db" | "heuristic";
}

/**
 * Resolve a model ID to its provider adapter factory.
 *
 * @param modelId - The model string from the request body (e.g. "gpt-4o", "claude-opus-4-6-20250918").
 * @returns ResolvedModel on success, null when the model cannot be mapped.
 */
export async function resolveModel(
  modelId: string,
): Promise<ResolvedModel | null> {
  // ── Stage 1: DB lookup ────────────────────────────────────────────────────
  try {
    const dbModel = await ModelModel.findByModelIdOnly(modelId);
    if (dbModel) {
      const factory = ADAPTER_FACTORIES[dbModel.provider];
      if (factory) {
        logger.debug(
          { modelId, provider: dbModel.provider },
          "[UnifiedResolver] Resolved via DB",
        );
        return {
          providerId: dbModel.provider,
          adapterFactory: factory,
          source: "db",
        };
      }

      // DB has the model but we don't have an adapter for its provider —
      // fall through to heuristic rather than failing silently.
      logger.warn(
        { modelId, provider: dbModel.provider },
        "[UnifiedResolver] DB match found but no adapter registered for provider",
      );
    }
  } catch (err) {
    // Non-fatal: DB errors must not block the heuristic path.
    logger.warn(
      { err, modelId },
      "[UnifiedResolver] DB lookup failed, falling back to heuristic",
    );
  }

  // ── Stage 2: Heuristic prefix matching ───────────────────────────────────
  const lowerModelId = modelId.toLowerCase();

  for (const [prefix, provider] of PREDETERMINED_PREFIXES) {
    if (
      lowerModelId.startsWith(prefix.toLowerCase()) ||
      modelId.startsWith(prefix) // preserve case for prefixes like "MiniMax-"
    ) {
      const factory = ADAPTER_FACTORIES[provider];
      if (factory) {
        logger.debug(
          { modelId, prefix, provider },
          "[UnifiedResolver] Resolved via heuristic prefix",
        );
        return {
          providerId: provider,
          adapterFactory: factory,
          source: "heuristic",
        };
      }
    }
  }

  logger.info(
    { modelId },
    "[UnifiedResolver] Could not map model to any known provider",
  );
  return null;
}
