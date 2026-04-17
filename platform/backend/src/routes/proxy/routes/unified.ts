/**
 * Unified LLM Gateway — /v1/unified/
 *
 * A single OpenAI-compatible entry point that auto-routes to the correct
 * upstream provider based on the `model` field.
 *
 * Endpoints:
 *   GET  /v1/unified/models
 *   POST /v1/unified/chat/completions           (default agent)
 *   POST /v1/unified/:agentId/chat/completions  (specific agent)
 *
 * All requests/responses conform to the OpenAI wire format.
 * Provider detection is performed by resolveModel() (DB-first, heuristic fallback).
 * Once the provider is known, the request is forwarded to handleLLMProxy()
 * with the matching adapter factory — inheriting all existing middleware:
 * auth, virtual API keys, TOON compression, dual-LLM, guardrails, observability, etc.
 */

import { RouteId } from "@shared";
import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import { z } from "zod";
import logger from "@/logging";
import { ModelModel } from "@/models";
import { constructResponseSchema, OpenAi, UuidIdSchema } from "@/types";
import { PROXY_API_PREFIX, PROXY_BODY_LIMIT } from "../common";
import { handleLLMProxy } from "../llm-proxy-handler";
import { resolveModel } from "./unified-model-resolver";

const UNIFIED_PREFIX = `${PROXY_API_PREFIX}/unified`;

const unifiedProxyRoutes: FastifyPluginAsyncZod = async (fastify) => {
  logger.info("[UnifiedProxy] Registering unified gateway routes");

  // ──────────────────────────────────────────────────────────────────────────
  // GET /v1/unified/models
  // Lists all chat-capable models from every configured provider in the
  // standard OpenAI { object: "list", data: [...] } format so that any
  // OpenAI-compatible SDK can enumerate available models.
  // ──────────────────────────────────────────────────────────────────────────
  fastify.get(
    `${UNIFIED_PREFIX}/models`,
    {
      schema: {
        operationId: RouteId.UnifiedListModels,
        description:
          "List all chat-capable models from all configured providers (OpenAI format)",
        tags: ["LLM Proxy"],
        response: constructResponseSchema(
          z.object({
            object: z.literal("list"),
            data: z.array(
              z.object({
                id: z.string(),
                object: z.literal("model"),
                created: z.number(),
                owned_by: z.string(),
              }),
            ),
          }),
        ),
      },
    },
    async (_request, reply) => {
      logger.debug("[UnifiedProxy] GET /models");

      const allModels = await ModelModel.findAll();
      const chatModels = allModels.filter((m) => ModelModel.supportsTextChat(m));

      const data = chatModels.map((m) => ({
        id: m.modelId,
        object: "model" as const,
        // Use lastSyncedAt as the created timestamp; fall back to epoch 0.
        created: m.lastSyncedAt
          ? Math.floor(new Date(m.lastSyncedAt).getTime() / 1000)
          : 0,
        owned_by: m.provider,
      }));

      logger.debug(
        { total: allModels.length, chatCapable: chatModels.length },
        "[UnifiedProxy] GET /models — returning models",
      );

      return reply.send({ object: "list", data });
    },
  );

  // ──────────────────────────────────────────────────────────────────────────
  // POST /v1/unified/chat/completions  (default agent)
  // ──────────────────────────────────────────────────────────────────────────
  fastify.post(
    `${UNIFIED_PREFIX}/chat/completions`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedChatCompletionsWithDefaultAgent,
        description:
          "OpenAI-compatible chat completion that auto-routes to the correct provider based on the model field (uses default agent)",
        tags: ["LLM Proxy"],
        body: OpenAi.API.ChatCompletionRequestSchema,
        headers: OpenAi.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          OpenAi.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const modelId = request.body.model;

      logger.debug(
        { modelId, url: request.url },
        "[UnifiedProxy] POST /chat/completions (default agent) — resolving provider",
      );

      const resolved = await resolveModel(modelId);

      if (!resolved) {
        logger.info(
          { modelId },
          "[UnifiedProxy] Unknown model — returning 400",
        );
        return reply.status(400).send({
          error: {
            message: `Model "${modelId}" could not be mapped to a known provider. Check the model ID or configure it in LLM Settings.`,
            type: "invalid_request_error",
            code: "unknown_model",
          },
        });
      }

      logger.info(
        { modelId, provider: resolved.providerId, source: resolved.source },
        "[UnifiedProxy] Routing to provider",
      );

      return handleLLMProxy(
        request.body,
        request,
        reply,
        resolved.adapterFactory,
      );
    },
  );

  // ──────────────────────────────────────────────────────────────────────────
  // POST /v1/unified/:agentId/chat/completions  (specific agent)
  // ──────────────────────────────────────────────────────────────────────────
  fastify.post(
    `${UNIFIED_PREFIX}/:agentId/chat/completions`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedChatCompletionsWithAgent,
        description:
          "OpenAI-compatible chat completion that auto-routes to the correct provider based on the model field (specific agent)",
        tags: ["LLM Proxy"],
        params: z.object({
          agentId: UuidIdSchema,
        }),
        body: OpenAi.API.ChatCompletionRequestSchema,
        headers: OpenAi.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          OpenAi.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const { agentId } = request.params;
      const modelId = request.body.model;

      logger.debug(
        { modelId, agentId, url: request.url },
        "[UnifiedProxy] POST /chat/completions (with agent) — resolving provider",
      );

      const resolved = await resolveModel(modelId);

      if (!resolved) {
        logger.info(
          { modelId, agentId },
          "[UnifiedProxy] Unknown model — returning 400",
        );
        return reply.status(400).send({
          error: {
            message: `Model "${modelId}" could not be mapped to a known provider. Check the model ID or configure it in LLM Settings.`,
            type: "invalid_request_error",
            code: "unknown_model",
          },
        });
      }

      logger.info(
        {
          modelId,
          agentId,
          provider: resolved.providerId,
          source: resolved.source,
        },
        "[UnifiedProxy] Routing to provider",
      );

      return handleLLMProxy(
        request.body,
        request,
        reply,
        resolved.adapterFactory,
      );
    },
  );
};

export default unifiedProxyRoutes;
