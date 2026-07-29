import axios from 'axios'

export const API_BASE_URL = (((import.meta as any).env?.VITE_API_BASE_URL as string | undefined) || '/api').replace(/\/$/, '')

const http = axios.create({ baseURL: API_BASE_URL, timeout: 120000 })

export async function getSummary() {
  const { data } = await http.get('/dashboard/summary')
  return data
}

export async function getHealth() {
  const { data } = await http.get('/health', { timeout: 8000 })
  return data
}

export async function getVehicles() {
  const { data } = await http.get('/vehicles')
  return data
}

export async function recommend(payload: any) {
  const { data } = await http.post('/recommend', payload)
  return data
}

export async function previewProfile(payload: any) {
  const { data } = await http.post('/profile/preview', payload)
  return data
}

export async function compare(payload: any) {
  const { data } = await http.post('/compare', payload)
  return data
}

export async function ragChat(query: string) {
  const { data } = await http.post('/rag/chat', { query, top_k: 6 })
  return data
}

export async function customerServiceChat(query: string, useWebSearch = true, history: any[] = []) {
  const { data } = await http.post('/customer-service/chat', { query, top_k: 6, use_web_search: useWebSearch, history })
  return data
}

export async function deepSearch(query: string) {
  const { data } = await http.post('/deep-search', { query, top_k: 6 })
  return data
}

export async function createLead(payload: any) {
  const { data } = await http.post('/leads', payload)
  return data
}

export async function getLeads() {
  const { data } = await http.get('/leads')
  return data
}

export async function publicConfig() {
  const { data } = await http.get('/config/public')
  return data
}

export async function checkLlmConfig() {
  const { data } = await http.get('/config/llm-check', { timeout: 30000 })
  return data
}

export async function getSystemReadiness() {
  const { data } = await http.get('/system/readiness')
  return data
}

export async function getReleaseGate() {
  const { data } = await http.get('/system/release-gate', { timeout: 120000 })
  return data
}

export async function runAcceptanceReport() {
  const { data } = await http.post('/system/acceptance-report', {}, { timeout: 120000 })
  return data
}

export async function generateDeliveryPackage() {
  const { data } = await http.post('/system/delivery-package', {}, { timeout: 120000 })
  return data
}

export async function rebuildRag() {
  const { data } = await http.post('/rag/rebuild')
  return data
}

export async function getKnowledgeFusionStatus() {
  const { data } = await http.get('/knowledge/fusion-status')
  return data
}

export async function getObsidianGraph() {
  const { data } = await http.get('/obsidian/graph')
  return data
}

export async function getRecommendationCases(limit = 20) {
  const { data } = await http.get('/obsidian/recommendation-cases', { params: { limit } })
  return data
}

export async function runRecommendationEvaluation() {
  const { data } = await http.post('/evaluation/recommendation')
  return data
}

export async function runAgentRegressionEvaluation() {
  const { data } = await http.post('/evaluation/agent-regression')
  return data
}

export async function createRecommendationFeedback(payload: any) {
  const { data } = await http.post('/recommendation-feedback', payload)
  return data
}

export async function getRecommendationFeedbackSummary() {
  const { data } = await http.get('/recommendation-feedback/summary')
  return data
}

export async function getRecommendationFeedbackReview() {
  const { data } = await http.get('/feedback/review')
  return data
}

export async function getOptimizationInsights() {
  const { data } = await http.get('/optimization/insights')
  return data
}

export async function getRealWorldOverview(limit = 30) {
  const { data } = await http.get('/real-world/overview', { params: { limit } })
  return data
}

export async function refreshRealWorldGovernance() {
  const { data } = await http.post('/real-world/governance')
  return data
}

export async function recommendRealWorld(payload: any) {
  const { data } = await http.post('/real-world/recommend', payload)
  return data
}

export async function getFusedCatalog(limitReal = 220) {
  const { data } = await http.get('/catalog/fused', { params: { limit_real: limitReal } })
  return data
}

export async function recommendFused(payload: any) {
  const { data } = await http.post('/recommend-fused', payload)
  return data
}

export async function recommendAgent(payload: any) {
  const { data } = await http.post('/agent/recommend', payload)
  return data
}

export async function seedObsidianProjectData() {
  const { data } = await http.post('/obsidian/seed-project-data')
  return data
}

export async function recommendStream(payload: any, callbacks: {
  onTrace: (entry: any) => void
  onResult: (result: any) => void
  onError: (error: string) => void
  onDone: () => void
}) {
  const response = await fetch(`${API_BASE_URL}/recommend-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.body) {
    callbacks.onError('当前浏览器不支持流式响应')
    return
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const part of parts) {
      const lines = part.split('\n')
      let eventType = ''
      let data = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7)
        if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (!data) continue
      try {
        const parsed = JSON.parse(data)
        if (eventType === 'trace') callbacks.onTrace(parsed)
        else if (eventType === 'result') callbacks.onResult(parsed)
        else if (eventType === 'error') callbacks.onError(parsed.error || String(parsed))
        else if (eventType === 'done') callbacks.onDone()
      } catch { /* skip malformed */ }
    }
  }
}

export async function clearRuntimeData() {
  const { data } = await http.post('/admin/clear-runtime-data')
  return data
}

export async function seedDemoData() {
  const { data } = await http.post('/demo/seed')
  return data
}
