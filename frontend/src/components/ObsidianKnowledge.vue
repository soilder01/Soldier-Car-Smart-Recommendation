<template>
  <section class="section">
    <div v-if="dataMode === 'fallback'" class="vault-warning">
      后端知识库接口暂时不可用，已自动切换到内置演示图谱；刷新或重启后端后会自动读取真实 Obsidian Vault。
    </div>
    <div class="kpi-grid">
      <div class="kpi"><span>Vault 节点</span><strong>{{ graph?.stats?.node_count || 0 }}</strong><p>Markdown 知识卡片</p></div>
      <div class="kpi"><span>双向链接</span><strong>{{ graph?.stats?.edge_count || 0 }}</strong><p>Obsidian 图谱连接</p></div>
      <div class="kpi"><span>RAG 总片段</span><strong>{{ fusion?.summary?.rag_chunks || 0 }}</strong><p>即时检索证据</p></div>
      <div class="kpi"><span>Vault 入库片段</span><strong>{{ fusion?.summary?.obsidian_chunks || 0 }}</strong><p>Obsidian 已进入 RAG</p></div>
      <div class="kpi"><span>联动状态</span><strong>{{ fusion?.summary?.linked ? '已联动' : '待联动' }}</strong><p>长期记忆 + 即时取证</p></div>
    </div>

    <div class="card fusion-card">
      <div class="card-title">
        <h3>RAG × Obsidian 联动闭环</h3>
        <el-button :loading="fusionLoading" type="primary" @click="refreshFusion">刷新联动状态</el-button>
      </div>
      <div class="fusion-flow">
        <div v-for="item in fusionRoles" :key="item.name" class="fusion-step">
          <span>{{ item.role }}</span>
          <b>{{ item.name }}</b>
          <p>{{ item.description }}</p>
        </div>
      </div>
      <div class="vault-tip">
        <b>当前模式：</b>{{ fusion?.summary?.mode || 'Obsidian长期记忆 + RAG即时检索' }}。Obsidian Vault 里的 Markdown 会被后端 RAG 索引读取，Agent 推荐时可从长期知识资产中即时取证。
      </div>
    </div>

    <div class="grid obsidian-layout">
      <div class="card">
        <div class="card-title">
          <h3>Obsidian 自生长知识库</h3>
          <div>
            <el-button @click="refreshGraph">刷新</el-button>
            <el-button type="primary" :loading="seeding" @click="seedData">抓取项目数据补充展示</el-button>
          </div>
        </div>
        <div class="vault-tip">
          <b>真实 Vault：</b>知识文件保存在项目的 <code>obsidian-vault/</code>，可直接用 Obsidian 打开，Graph View 会识别这些 <code>[[双向链接]]</code>。
        </div>
        <div class="graph-board">
          <svg viewBox="0 0 760 420" role="img" aria-label="Obsidian 知识图谱">
            <line v-for="edge in visibleEdges" :key="`${edge.source}-${edge.target}`" :x1="point(edge.source).x" :y1="point(edge.source).y" :x2="point(edge.target).x" :y2="point(edge.target).y" />
            <g v-for="node in visibleNodes" :key="node.id" class="graph-node" @click="selectedId = node.id">
              <circle :cx="point(node.id).x" :cy="point(node.id).y" :r="selectedId === node.id ? 18 : 13" :class="node.type" />
              <text :x="point(node.id).x + 18" :y="point(node.id).y + 5">{{ node.title }}</text>
            </g>
          </svg>
        </div>
      </div>

      <div class="card">
        <h3>节点详情</h3>
        <div v-if="selectedNode" class="node-detail">
          <div class="node-path">{{ selectedNode.path }}</div>
          <h2>{{ selectedNode.title }}</h2>
          <p>{{ selectedNode.excerpt }}</p>
          <div class="tags"><span v-for="tag in selectedNode.tags" :key="tag">{{ tag }}</span></div>
          <div class="link-list">
            <b>关联节点</b>
            <button v-for="link in selectedNode.links" :key="link" @click="selectByTitle(link)">{{ link }}</button>
            <span v-if="!selectedNode.links.length" class="muted">暂无外链</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card case-board">
      <div class="card-title">
        <h3>推荐案例沉淀</h3>
        <el-input v-model="caseKeyword" clearable placeholder="搜索车型、画像或路径" class="case-search" />
      </div>
      <div v-if="filteredCases.length" class="case-grid">
        <button v-for="item in filteredCases" :key="item.id" class="case-card" @click="selectedId = item.id">
          <span>{{ item.updated_at }}</span>
          <b>{{ item.title }}</b>
          <p>{{ item.excerpt }}</p>
          <small>{{ item.path }}</small>
        </button>
      </div>
      <div v-else class="empty-case">
        暂无推荐案例。去“智能推荐”页生成一次推荐后，系统会自动把画像、推荐车型和报告摘要写入 Obsidian Vault。
      </div>
    </div>

    <div class="grid two">
      <div class="card">
        <h3>知识节点列表</h3>
        <el-table :data="filteredNodes" height="430" @row-click="handleRowClick">
          <el-table-column prop="title" label="标题" min-width="180" />
          <el-table-column prop="type" label="类型" width="120" />
          <el-table-column prop="updated_at" label="更新时间" width="170" />
        </el-table>
      </div>
      <div class="card">
        <h3>标签分布</h3>
        <div class="tag-cloud">
          <button v-for="item in tagItems" :key="item.name" :class="{ active: tagFilter === item.name }" @click="tagFilter = tagFilter === item.name ? '' : item.name">
            {{ item.name }} <span>{{ item.value }}</span>
          </button>
        </div>
        <h3 class="mt">类型分布</h3>
        <div class="type-list">
          <div v-for="item in typeItems" :key="item.name"><span>{{ item.name }}</span><b>{{ item.value }}</b></div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { API_BASE_URL, getKnowledgeFusionStatus, getObsidianGraph, getRecommendationCases, seedObsidianProjectData } from '../api/client'

const fallbackNodes = [
  node('home', 'Soldier Car 自生长知识库首页', 'index', ['项目总览', 'Obsidian'], ['车型库总览', '用户画像解析优化', '推荐链路', '自生长知识库方案'], '00-入口/Soldier-Car-知识库首页.md'),
  node('vehicles', '车型库总览', 'vehicle-knowledge', ['车型库', '新能源汽车'], ['推荐链路', '用户画像解析优化'], '01-车型知识/车型库总览.md'),
  node('profile', '用户画像解析优化', 'optimization', ['用户画像', 'LLM解析'], ['推荐链路', '车型库总览'], '02-用户画像/用户画像解析优化.md'),
  node('recommend', '推荐链路', 'system-flow', ['推荐系统', 'MultiAgent'], ['用户画像解析优化', 'RAG知识库优化'], '03-推荐系统/推荐链路.md'),
  node('rag', 'RAG知识库优化', 'rag-source', ['RAG', '知识库'], ['推荐链路', '自生长知识库方案'], '04-RAG知识库/RAG知识库优化.md'),
  node('obsidian', '自生长知识库方案', 'architecture', ['Obsidian', '自生长知识库'], ['Soldier Car 自生长知识库首页', 'RAG知识库优化'], '05-自生长机制/自生长知识库方案.md'),
  node('model-y', '特斯拉 Model Y', 'vehicle', ['车型', '纯电', 'SUV'], ['车型库总览', '推荐链路'], '01-车型知识/特斯拉-Model-Y.md'),
  node('xpeng-g6', '小鹏 G6', 'vehicle', ['车型', '纯电', 'SUV'], ['车型库总览', '推荐链路'], '01-车型知识/小鹏-G6.md'),
  node('scenario-family', '三口之家城市通勤选车', 'scenario', ['家庭用车', '城市通勤'], ['用户画像解析优化', '推荐链路'], '07-测试样例/三口之家城市通勤选车.md'),
  node('case-family', '推荐案例-三口之家-理想L6', 'recommendation-case', ['推荐案例', '用户画像', '自生长知识库'], ['用户画像解析优化', '推荐链路', '理想 L6'], '08-推荐案例/推荐案例-三口之家-理想L6.md'),
]

function node(id: string, title: string, type: string, tags: string[], links: string[], path: string) {
  return { id, title, type, tags, links, path, updated_at: '2026-06-22 12:00:00', excerpt: `${title} 是当前新能源汽车智能推荐项目沉淀到 Obsidian Vault 的知识节点，用于展示双向链接、标签和自生长关系。` }
}

function buildGraph(nodes: any[]) {
  const safeNodes = (nodes || []).map(item => ({ ...item, links: item.links || [], tags: item.tags || [] }))
  const titleToId: Record<string, string> = Object.fromEntries(safeNodes.map(item => [item.title, item.id]))
  const edges = safeNodes.flatMap(item => item.links.map((link: string) => ({ source: item.id, target: titleToId[link], label: link }))).filter((edge: any) => edge.target)
  const tags: Record<string, number> = {}
  const types: Record<string, number> = {}
  safeNodes.forEach(item => {
    types[item.type] = (types[item.type] || 0) + 1
    item.tags.forEach((tag: string) => { tags[tag] = (tags[tag] || 0) + 1 })
  })
  return { stats: { node_count: safeNodes.length, edge_count: edges.length, tag_count: Object.keys(tags).length, vehicle_node_count: safeNodes.filter(item => item.type === 'vehicle').length, recommendation_case_count: safeNodes.filter(item => item.type === 'recommendation-case').length }, nodes: safeNodes, edges, tag_distribution: tags, type_distribution: types }
}

const fallbackGraph = buildGraph(fallbackNodes)
const fallbackCases = fallbackGraph.nodes.filter((item: any) => item.type === 'recommendation-case')
const graph = ref<any>(fallbackGraph)
const recommendationCases = ref<any[]>(fallbackCases)
const fusion = ref<any>({
  summary: { mode: 'Obsidian长期记忆 + RAG即时检索', rag_chunks: 0, obsidian_chunks: 0, linked: false },
  roles: [
    { name: 'Obsidian', role: '长期知识库', description: '沉淀推荐案例、反馈复盘、测试评估、治理报告和交付报告' },
    { name: 'RAG', role: '即时取证', description: '从知识库与 Obsidian Vault 中检索相关片段，支撑 Agent 推荐解释' },
    { name: 'Agent', role: '推理决策', description: '结合画像、车型数据、RAG证据和反馈策略生成推荐' },
  ],
})
const selectedId = ref('home')
const tagFilter = ref('')
const caseKeyword = ref('')
const seeding = ref(false)
const fusionLoading = ref(false)
const dataMode = ref<'api' | 'fallback'>('fallback')

const nodes = computed(() => graph.value?.nodes || [])
const edges = computed(() => graph.value?.edges || [])
const filteredNodes = computed(() => tagFilter.value ? nodes.value.filter((item: any) => item.tags?.includes(tagFilter.value)) : nodes.value)
const filteredCases = computed(() => {
  const keyword = caseKeyword.value.trim().toLowerCase()
  if (!keyword) return recommendationCases.value
  return recommendationCases.value.filter((item: any) => `${item.title} ${item.excerpt} ${item.path}`.toLowerCase().includes(keyword))
})
const visibleNodes = computed(() => filteredNodes.value.slice(0, 18))
const visibleIds = computed(() => new Set(visibleNodes.value.map((item: any) => item.id)))
const visibleEdges = computed(() => edges.value.filter((edge: any) => visibleIds.value.has(edge.source) && visibleIds.value.has(edge.target)).slice(0, 40))
const selectedNode = computed(() => nodes.value.find((item: any) => item.id === selectedId.value) || nodes.value[0])
const tagItems = computed(() => Object.entries(graph.value?.tag_distribution || {}).map(([name, value]) => ({ name, value })))
const typeItems = computed(() => Object.entries(graph.value?.type_distribution || {}).map(([name, value]) => ({ name, value })))
const fusionRoles = computed(() => fusion.value?.roles || [])

function point(id: string) {
  const index = visibleNodes.value.findIndex((item: any) => item.id === id)
  const total = Math.max(visibleNodes.value.length, 1)
  const angle = (Math.PI * 2 * Math.max(index, 0)) / total
  const radius = index % 3 === 0 ? 150 : index % 3 === 1 ? 120 : 180
  return { x: 380 + Math.cos(angle) * radius, y: 210 + Math.sin(angle) * radius }
}

function selectByTitle(title: string) {
  const target = nodes.value.find((item: any) => item.title === title)
  if (target) selectedId.value = target.id
}

function handleRowClick(row: any) {
  selectedId.value = row.id
}

async function refreshFusion() {
  fusionLoading.value = true
  try {
    if (API_BASE_URL === '/api') throw new Error('静态预览启用内置联动状态')
    fusion.value = await getKnowledgeFusionStatus()
  } catch {
    fusion.value = {
      ...fusion.value,
      summary: {
        ...fusion.value.summary,
        rag_chunks: graph.value?.stats?.node_count || 0,
        obsidian_chunks: graph.value?.stats?.node_count || 0,
        linked: true,
      },
    }
  } finally {
    fusionLoading.value = false
  }
}

async function loadRecommendationCases() {
  if (API_BASE_URL === '/api') {
    recommendationCases.value = fallbackCases
    return
  }
  const res = await getRecommendationCases(30)
  recommendationCases.value = Array.isArray(res?.cases) ? res.cases.map((item: any) => ({ ...item, links: item.links || [], tags: item.tags || [] })) : []
}

async function refreshGraph() {
  try {
    if (API_BASE_URL === '/api') throw new Error('静态预览启用内置图谱')
    const data = await getObsidianGraph()
    graph.value = buildGraph(data?.nodes || [])
    await loadRecommendationCases()
    await refreshFusion()
    dataMode.value = 'api'
    if (!selectedId.value && nodes.value.length) selectedId.value = nodes.value[0].id
  } catch {
    graph.value = fallbackGraph
    recommendationCases.value = fallbackCases
    selectedId.value = 'home'
    dataMode.value = 'fallback'
  }
}

async function seedData() {
  seeding.value = true
  try {
    if (API_BASE_URL === '/api') throw new Error('静态预览启用内置图谱')
    const res = await seedObsidianProjectData()
    graph.value = buildGraph(res.graph?.nodes || [])
    await loadRecommendationCases()
    dataMode.value = 'api'
    selectedId.value = graph.value?.nodes?.[0]?.id || ''
    ElMessage.success(`已补充 ${res.created_count} 个 Obsidian 节点`)
  } catch {
    graph.value = fallbackGraph
    recommendationCases.value = fallbackCases
    dataMode.value = 'fallback'
    ElMessage.warning('后端暂不可用，已展示内置演示图谱')
  } finally {
    seeding.value = false
  }
}

onMounted(() => {
  refreshGraph()
  refreshFusion()
})
</script>
