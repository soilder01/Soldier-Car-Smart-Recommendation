<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="logo">S</div>
        <div>
          <h1>Soldier</h1>
          <p>智能推荐中台</p>
        </div>
      </div>
      <button v-for="item in navs" :key="item.key" class="nav-item" :class="{ active: active === item.key }" @click="setActive(item.key)">
        <span>{{ item.icon }}</span>
        <b>{{ item.label }}</b>
      </button>
    </aside>

    <main class="main">
      <header class="hero">
        <div>
          <p class="eyebrow">Multi-Agent · RAG · DeepSearch · Skills · SQLite</p>
          <h2>{{ currentTitle }}</h2>
          <p>{{ currentSubtitle }}</p>
        </div>
        <div class="hero-actions">
          <div class="backend-status" :class="backendStatus.status">
            <span></span>
            <div><b>{{ backendStatusText }}</b><small>{{ backendStatusDetail }}</small></div>
          </div>
          <el-button type="primary" size="large" @click="runDemo">一键生成推荐</el-button>
        </div>
      </header>

      <section v-if="active === 'dashboard'" class="section">
        <div class="kpi-grid">
          <div class="kpi"><span>车型库</span><strong>{{ summary?.vehicle_count || 0 }}</strong><p>覆盖主流新能源车型</p></div>
          <div class="kpi"><span>推荐次数</span><strong>{{ summary?.recommendation_count || 0 }}</strong><p>推荐日志自动沉淀</p></div>
          <div class="kpi"><span>平均预算</span><strong>{{ money(summary?.avg_budget || 0) }}</strong><p>线索预算中枢</p></div>
          <div class="kpi"><span>知识片段</span><strong>{{ summary?.rag_stats?.chunks || 0 }}</strong><p>RAG 可追溯证据</p></div>
        </div>

        <div class="grid three">
          <div class="card"><h3>能源类型分布</h3><VChart class="chart" :option="energyOption" autoresize /></div>
          <div class="card"><h3>预算分布</h3><VChart class="chart" :option="budgetOption" autoresize /></div>
          <div class="card"><h3>客户关注点</h3><VChart class="chart" :option="concernOption" autoresize /></div>
        </div>
        <div class="grid two">
          <div class="card"><h3>热门车型销量 Top</h3><VChart class="chart tall" :option="hotModelOption" autoresize /></div>
          <div class="card"><h3>价格 - 续航散点</h3><VChart class="chart tall" :option="scatterOption" autoresize /></div>
        </div>
      </section>

      <section v-if="active === 'evaluation'" class="section">
        <div class="kpi-grid">
          <div class="kpi"><span>测试用例</span><strong>{{ evaluationSummary.case_count || 0 }}</strong><p>固定推荐回归集</p></div>
          <div class="kpi"><span>通过率</span><strong>{{ evaluationSummary.pass_rate || 0 }}%</strong><p>Pass 用例占比</p></div>
          <div class="kpi"><span>平均分</span><strong>{{ evaluationSummary.average_score || 0 }}</strong><p>画像与 Top 推荐综合分</p></div>
          <div class="kpi"><span>异常用例</span><strong>{{ (evaluationSummary.warned || 0) + (evaluationSummary.failed || 0) }}</strong><p>需继续优化</p></div>
        </div>
        <div class="grid evaluation-layout">
          <div class="card">
            <div class="card-title">
              <h3>推荐质量评估闭环</h3>
              <el-button type="primary" :loading="evaluationLoading" @click="runEvaluation">运行评估</el-button>
            </div>
            <div class="evaluation-note" v-if="evaluationNote?.path">
              <b>已写入 Obsidian 评估报告</b><span>{{ evaluationNote.title }} · {{ evaluationNote.path }}</span>
            </div>
            <el-table :data="evaluationCases" height="520">
              <el-table-column prop="name" label="用例" min-width="150" />
              <el-table-column label="状态" width="100">
                <template #default="{ row }"><el-tag :type="row.status === 'pass' ? 'success' : row.status === 'warn' ? 'warning' : 'danger'">{{ row.status }}</el-tag></template>
              </el-table-column>
              <el-table-column prop="score" label="分数" width="90" />
              <el-table-column label="Top 推荐" min-width="210">
                <template #default="{ row }">{{ row.top_models?.slice(0, 3).join('、') }}</template>
              </el-table-column>
              <el-table-column prop="diagnosis" label="诊断" min-width="220" />
            </el-table>
          </div>
          <div class="card">
            <h3>用例得分分布</h3>
            <VChart class="chart" :option="evaluationOption" autoresize />
            <h3 class="mt">失败/警告检查项</h3>
            <div class="eval-issues">
              <div v-for="item in evaluationIssues" :key="item.key"><b>{{ item.caseName }}</b><p>{{ item.message }}</p><small>期望：{{ item.expected }}；实际：{{ item.actual }}</small></div>
              <div v-if="!evaluationIssues.length" class="empty-case">当前固定用例全部通过，后续应继续补充真实失败案例。</div>
            </div>
          </div>
        </div>
        <div class="card agent-regression-card">
          <div class="card-title">
            <h3>Agent 端到端回归评估</h3>
            <span class="saved-pill">阶段 G2</span>
          </div>
          <div class="evaluation-note" v-if="agentRegression.obsidian_note?.path">
            <b>已写入 Obsidian 回归报告</b><span>{{ agentRegression.obsidian_note.title }} · {{ agentRegression.obsidian_note.path }}</span>
          </div>
          <div class="regression-kpis">
            <div><span>回归用例</span><b>{{ agentRegression.summary?.case_count || 0 }}</b></div>
            <div><span>通过率</span><b>{{ agentRegression.summary?.pass_rate || 0 }}%</b></div>
            <div><span>平均分</span><b>{{ agentRegression.summary?.average_score || 0 }}</b></div>
            <div><span>异常</span><b>{{ (agentRegression.summary?.warned || 0) + (agentRegression.summary?.failed || 0) }}</b></div>
          </div>
          <el-table :data="agentRegression.cases || []" height="300">
            <el-table-column prop="name" label="Agent 用例" min-width="170" />
            <el-table-column prop="selected_pool" label="候选池" width="100" />
            <el-table-column prop="score" label="分数" width="90" />
            <el-table-column label="Trace 覆盖" min-width="240">
              <template #default="{ row }">{{ row.trace_agents?.join(' → ') }}</template>
            </el-table-column>
            <el-table-column label="策略规则" width="100">
              <template #default="{ row }">{{ row.feedback_policy_rules?.length || 0 }} 条</template>
            </el-table-column>
          </el-table>
        </div>
      </section>

      <section v-if="active === 'feedback'" class="section">
        <div class="kpi-grid">
          <div class="kpi"><span>反馈总量</span><strong>{{ feedbackSummary.total || 0 }}</strong><p>推荐卡片人工反馈</p></div>
          <div class="kpi"><span>正反馈率</span><strong>{{ feedbackSummary.positive_rate || 0 }}%</strong><p>推荐被认可比例</p></div>
          <div class="kpi"><span>正反馈</span><strong>{{ feedbackSummary.positive || 0 }}</strong><p>推荐准确/可采纳</p></div>
          <div class="kpi"><span>负反馈</span><strong>{{ feedbackSummary.negative || 0 }}</strong><p>需要优化样本</p></div>
        </div>
        <div class="grid feedback-layout">
          <div class="card">
            <div class="card-title"><h3>反馈车型分布</h3><el-button @click="refreshFeedback">刷新反馈</el-button></div>
            <VChart class="chart" :option="feedbackOption" autoresize />
            <h3 class="mt">反馈原因 Top</h3>
            <div class="type-list">
              <div v-for="item in feedbackSummary.reasons || []" :key="item.reason"><span>{{ item.reason }}</span><b>{{ item.count }}</b></div>
            </div>
          </div>
          <div class="card">
            <h3>最近反馈</h3>
            <el-table :data="feedbackSummary.recent || []" height="520">
              <el-table-column prop="created_at" label="时间" width="170" />
              <el-table-column prop="model_name" label="车型" width="150" />
              <el-table-column label="反馈" width="100">
                <template #default="{ row }"><el-tag :type="row.rating === 'positive' ? 'success' : row.rating === 'negative' ? 'danger' : 'info'">{{ row.rating }}</el-tag></template>
              </el-table-column>
              <el-table-column prop="candidate_pool" label="候选池" width="100" />
              <el-table-column prop="reason" label="原因" />
            </el-table>
          </div>
        </div>
        <div class="grid feedback-review-layout">
          <div class="card">
            <h3>候选池质量复盘</h3>
            <div class="pool-quality-list">
              <div v-for="item in feedbackSummary.pool_rows || []" :key="item.candidate_pool">
                <b>{{ item.candidate_pool }}</b>
                <span>{{ item.total }} 条反馈 · 正反馈率 {{ item.positive_rate }}%</span>
                <el-progress :percentage="item.positive_rate || 0" :stroke-width="10" />
              </div>
              <div v-if="!(feedbackSummary.pool_rows || []).length" class="empty-case">暂无候选池维度反馈，推荐卡片提交反馈后会自动聚合。</div>
            </div>
          </div>
          <div class="card">
            <h3>场景风险 Top</h3>
            <div class="type-list">
              <div v-for="item in feedbackSummary.scene_rows || []" :key="item.scenario"><span>{{ item.scenario }} · {{ item.total }} 条</span><b>{{ item.negative_rate }}%</b></div>
            </div>
            <p class="muted mt">负反馈率越高，Agent 后续越需要补强该场景的风险提示和候选对比。</p>
          </div>
          <div class="card">
            <div class="card-title"><h3>Agent 复盘结论</h3><span v-if="feedbackReview.obsidian_note?.path" class="saved-pill">已写入 Obsidian</span></div>
            <div class="review-list">
              <div v-for="item in feedbackReview.insights || []" :key="item.title">
                <b>{{ item.title }}</b>
                <p>{{ item.evidence }}</p>
                <small>{{ item.action }}</small>
              </div>
              <div v-if="!(feedbackReview.insights || []).length" class="empty-case">暂无复盘结论，先积累反馈样本。</div>
            </div>
          </div>
        </div>
      </section>

      <section v-if="active === 'optimization'" class="section">
        <div class="kpi-grid">
          <div class="kpi"><span>建议数</span><strong>{{ optimization.summary?.item_count || 0 }}</strong><p>自动生成优化任务</p></div>
          <div class="kpi"><span>高优先级</span><strong>{{ optimization.summary?.p1_count || 0 }}</strong><p>P0/P1 需要优先处理</p></div>
          <div class="kpi"><span>反馈样本</span><strong>{{ optimization.summary?.feedback_total || 0 }}</strong><p>人工反馈来源</p></div>
          <div class="kpi"><span>评估通过率</span><strong>{{ optimization.summary?.evaluation_pass_rate || 0 }}%</strong><p>固定用例健康度</p></div>
        </div>
        <div class="card">
          <div class="card-title">
            <h3>自动优化建议中心</h3>
            <el-button type="primary" :loading="optimizationLoading" @click="refreshOptimization">刷新建议</el-button>
          </div>
          <div class="evaluation-note" v-if="optimization.obsidian_note?.path">
            <b>已写入 Obsidian 优化记录</b><span>{{ optimization.obsidian_note.title }} · {{ optimization.obsidian_note.path }}</span>
          </div>
          <div class="optimization-list">
            <div v-for="item in optimization.items || []" :key="item.title" class="optimization-card" :class="item.priority.toLowerCase()">
              <span>{{ item.priority }} · {{ item.source }}</span>
              <b>{{ item.title }}</b>
              <p>{{ item.evidence }}</p>
              <small>{{ item.action }}</small>
            </div>
            <div v-if="!(optimization.items || []).length" class="empty-case">暂无优化建议，建议继续补充真实反馈和边界测试用例。</div>
          </div>
        </div>
      </section>

      <section v-if="active === 'realdata'" class="section">
        <div class="kpi-grid">
          <div class="kpi"><span>真实样本</span><strong>{{ realWorld.stats?.record_count || realWorld.quality?.record_count || 0 }}</strong><p>已沉淀新能源车型规格</p></div>
          <div class="kpi"><span>品牌覆盖</span><strong>{{ realWorld.quality?.unique_brand_count || 0 }}</strong><p>真实公开数据品牌数</p></div>
          <div class="kpi"><span>真实评估</span><strong>{{ realWorld.evaluation?.pass_rate || 0 }}%</strong><p>{{ realWorld.evaluation?.passed_count || 0 }}/{{ realWorld.evaluation?.case_count || 0 }} 用例通过</p></div>
          <div class="kpi"><span>治理评分</span><strong>{{ realWorld.governance?.summary?.quality_score || 0 }}</strong><p>数据质量治理分</p></div>
        </div>
        <div class="grid real-data-layout">
          <div class="card">
            <div class="card-title"><h3>真实数据源与质量</h3><el-button :loading="realWorldLoading" type="primary" @click="refreshRealWorld">刷新真实数据</el-button></div>
            <div class="real-source"><b>{{ realWorld.quality?.source }}</b><span>年份范围：{{ realWorld.quality?.year_range?.join(' - ') }} · 平均续航 {{ realWorld.stats?.avg_range_km || 0 }}km · 平均电池 {{ realWorld.stats?.avg_battery_kwh || 0 }}kWh</span></div>
            <div class="grid two compact">
              <div><h3>品牌样本分布</h3><VChart class="chart" :option="realBrandOption" autoresize /></div>
              <div><h3>车型类型分布</h3><VChart class="chart" :option="realTypeOption" autoresize /></div>
            </div>
            <div class="enrichment-strip">
              <div><span>补齐后均值质量</span><b>{{ realWorld.enrichment?.avg_data_quality_score || 0 }}分</b></div>
              <div><span>补齐输出</span><b>{{ realWorld.enrichment?.record_count || realWorld.quality?.record_count || 0 }}条</b></div>
              <div><span>融合候选池</span><b>{{ fusedCatalog.summary?.total || 0 }}条</b></div>
              <div><span>真实扩展入池</span><b>{{ fusedCatalog.summary?.real_count || 0 }}条</b></div>
            </div>
            <div class="grid two compact">
              <div><h3>原始字段缺口 Top</h3><el-table :data="realMissingRows" height="220"><el-table-column prop="field" label="字段" /><el-table-column prop="count" label="缺失" width="90" /></el-table></div>
              <div><h3>规则补齐字段</h3><el-table :data="realEstimatedRows" height="220"><el-table-column prop="field" label="字段" /><el-table-column prop="count" label="补齐" width="90" /></el-table></div>
            </div>
          </div>
          <div class="card">
            <h3>真实推荐评估结果</h3>
            <div class="real-eval-list">
              <div v-for="item in realEvaluationRows" :key="item.name" class="real-eval-card" :class="{ pass: item.passed }">
                <span>{{ item.passed ? '通过' : '需优化' }}</span>
                <b>{{ item.name }}</b>
                <p v-for="car in item.top_recommendations?.slice(0, 3) || []" :key="`${item.name}-${car.brand}-${car.model}`">{{ car.brand }} {{ car.model }} · {{ car.type }} · {{ car.range }}km · {{ car.score }}分</p>
              </div>
              <div v-if="!realEvaluationRows.length" class="empty-case">暂无后端真实评估详情，已展示本地数据概况。</div>
            </div>
          </div>
        </div>
        <div class="card data-governance-card">
          <div class="card-title"><h3>真实数据治理中心</h3><el-button :loading="realWorldLoading" @click="runRealWorldGovernance">重新治理</el-button></div>
          <div class="regression-kpis">
            <div><span>重复组</span><b>{{ realWorld.governance?.summary?.duplicate_group_count || 0 }}</b></div>
            <div><span>异常记录</span><b>{{ realWorld.governance?.summary?.anomaly_record_count || 0 }}</b></div>
            <div><span>可信来源</span><b>{{ realWorld.governance?.summary?.trusted_source_rate || 0 }}%</b></div>
            <div><span>缺失字段</span><b>{{ realWorld.governance?.summary?.missing_field_count || 0 }}</b></div>
          </div>
          <div class="grid three compact">
            <div><h3>治理动作</h3><div class="governance-list"><div v-for="item in realWorld.governance?.actions || []" :key="item.title"><b>{{ item.priority }} · {{ item.title }}</b><p>{{ item.evidence }}</p><small>{{ item.action }}</small></div></div></div>
            <div><h3>重复车型 Top</h3><el-table :data="realWorld.governance?.duplicates || []" height="260"><el-table-column prop="brand" label="品牌" /><el-table-column prop="model" label="车型" /><el-table-column prop="count" label="重复" width="80" /></el-table></div>
            <div><h3>异常参数 Top</h3><div class="governance-list"><div v-for="item in realWorld.governance?.anomalies?.slice(0, 6) || []" :key="`${item.brand}-${item.model}-${item.model_year}`"><b>{{ item.brand }} {{ item.model }}</b><p>{{ item.issues?.map((x: any) => x.message).join('；') }}</p></div></div></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title"><h3>真实候选推荐排序</h3><el-button type="primary" :loading="realWorldRecLoading" @click="runRealWorldRecommend">用当前画像跑真实候选</el-button></div>
          <el-table :data="realWorldRecs" height="360">
            <el-table-column label="车型" min-width="190"><template #default="{ row }"><b>{{ row.brand }} {{ row.model }}</b></template></el-table-column>
            <el-table-column prop="score" label="推荐分" width="90" />
            <el-table-column prop="vehicle_type" label="类型" width="90" />
            <el-table-column label="价格" width="120"><template #default="{ row }">{{ Math.round(row.price_min / 10000) }}-{{ Math.round(row.price_max / 10000) }}万</template></el-table-column>
            <el-table-column prop="cltc_range" label="续航km" width="100" />
            <el-table-column prop="data_quality_score" label="数据质量" width="110" />
            <el-table-column label="说明" min-width="260"><template #default="{ row }">{{ row.highlights || row.reasons?.slice(0, 2).join('；') }}</template></el-table-column>
          </el-table>
        </div>
        <div class="card">
          <div class="card-title"><h3>融合候选池推荐排序</h3><el-button type="success" :loading="fusedLoading" @click="runFusedRecommend">本地+真实融合推荐</el-button></div>
          <el-table :data="fusedRecs" height="320">
            <el-table-column label="车型" min-width="190"><template #default="{ row }"><b>{{ row.brand }} {{ row.model }}</b></template></el-table-column>
            <el-table-column prop="score" label="推荐分" width="90" />
            <el-table-column label="来源" width="130"><template #default="{ row }"><el-tag :type="row.catalog_source === 'local_curated' ? 'success' : 'warning'">{{ row.catalog_source === 'local_curated' ? '本地精选' : '真实扩展' }}</el-tag></template></el-table-column>
            <el-table-column prop="vehicle_type" label="类型" width="90" />
            <el-table-column prop="cltc_range" label="续航km" width="100" />
            <el-table-column label="理由" min-width="260"><template #default="{ row }">{{ row.reasons?.slice(0, 2).join('；') }}</template></el-table-column>
          </el-table>
        </div>
        <div class="card">
          <h3>真实样本预览</h3>
          <el-table :data="realWorld.samples || []" height="420">
            <el-table-column prop="brand" label="品牌" width="110" />
            <el-table-column prop="model" label="车型" min-width="160" />
            <el-table-column prop="model_year" label="年份" width="90" />
            <el-table-column prop="vehicle_type" label="类型" width="90" />
            <el-table-column prop="range_km" label="续航km" width="100" />
            <el-table-column prop="battery_kwh" label="电池kWh" width="110" />
            <el-table-column prop="dc_charge_kw" label="快充kW" width="100" />
            <el-table-column label="来源" min-width="220"><template #default="{ row }"><a class="source-link" :href="normalizeUrl(row.source_url)" target="_blank">{{ row.source_url }}</a></template></el-table-column>
          </el-table>
        </div>
      </section>

      <section v-if="active === 'recommend'" class="section">
        <div class="agent-workbench">
          <div class="workbench-main-card">
            <span class="workbench-label">Agent 工作台主入口</span>
            <h3>一句需求 → 画像解析 → 候选池选择 → 推荐排序 → 可解释报告 → Obsidian 记忆 → 反馈复盘</h3>
            <p>辅助页面已收束为证据、评估和复盘入口；日常使用优先从这里生成推荐并查看完整 Agent 状态。</p>
          </div>
          <div class="workbench-status-grid">
            <div v-for="item in agentWorkspaceStats" :key="item.label" class="workbench-status-card">
              <span>{{ item.label }}</span>
              <b>{{ item.value }}</b>
              <small>{{ item.desc }}</small>
            </div>
          </div>
        </div>
        <div class="card auxiliary-hub">
          <div class="card-title"><h3>辅助能力入口</h3><span class="saved-pill">二级视图</span></div>
          <div class="auxiliary-grid">
            <button v-for="item in auxiliaryViews" :key="item.key" @click="setActive(item.key)">
              <b>{{ item.title }}</b>
              <span>{{ item.desc }}</span>
            </button>
          </div>
        </div>
        <div class="grid recommend-layout">
          <div class="card">
            <div class="card-title"><h3>客户购车需求</h3><div class="title-actions"><el-button :loading="profileParsingLoading" @click="parseProfile">解析画像</el-button><el-button type="primary" :loading="loading" @click="submitRecommend">生成推荐</el-button></div></div>
            <label class="field wide">
              <span>自然语言需求</span>
              <el-input v-model="query" type="textarea" :rows="5" />
            </label>
            <div class="form-grid">
              <label class="field"><span>预算上限（元）</span><el-input-number v-model="profile.budget_max" :min="50000" :step="10000" placeholder="预算上限" /></label>
              <label class="field"><span>偏好车型</span><el-select v-model="profile.preferred_type" placeholder="偏好车型">
                  <el-option label="不限" value="" />
                  <el-option label="SUV" value="SUV" />
                  <el-option label="轿车" value="轿车" />
                  <el-option label="MPV" value="MPV" />
                  <el-option label="旅行车" value="旅行车" />
                  <el-option label="跑车" value="跑车" />
                  <el-option label="轿跑" value="轿跑" />
                  <el-option label="豪车" value="豪车" />
                </el-select></label>
            </div>
            <div class="switch-row">
              <div>
                <b>DeepSearch 联网增强</b>
                <p>开启后，智能推荐会调用 Web Search + RAG 补充公开资料，适合复杂选车和竞品问题。</p>
              </div>
              <el-switch v-model="useDeepSearch" active-text="开启" inactive-text="关闭" />
            </div>
            <div class="switch-row candidate-row">
              <div>
                <b>Agent 候选池策略</b>
                <p>默认由后端 Agent 自动选择；也可指定本地精选、真实扩展或融合池，前端不再绕过后端编排。</p>
              </div>
              <el-segmented v-model="candidatePool" :options="candidatePoolOptions" />
            </div>
            <div class="profile-panel" v-if="profilePreview">
              <div class="profile-head">
                <div>
                  <b>系统理解到的用户画像</b>
                  <p>{{ profilePreview.summary }}</p>
                </div>
                <el-progress type="circle" :width="58" :percentage="profilePreview.confidence" />
              </div>
              <div class="profile-fields">
                <div v-for="field in profilePreview.fields" :key="field.field" :class="['profile-field', { empty: !field.detected }]">
                  <span>{{ field.label }}</span>
                  <b>{{ field.display }}</b>
                  <small>{{ field.source }}</small>
                </div>
              </div>
              <div class="profile-insights">
                <p v-for="item in profilePreview.insights" :key="item">{{ item }}</p>
              </div>
              <div class="profile-missing" v-if="profilePreview.missing_fields?.length">
                待补充：{{ profilePreview.missing_fields.join('、') }}
              </div>
            </div>
          </div>

          <div class="card">
            <h3>推荐分项雷达</h3>
            <VChart class="chart" :option="radarOption" autoresize />
          </div>
        </div>

        <div class="recommend-cards">
          <div v-for="item in recommendations" :key="item.id" class="vehicle-card">
            <div class="card-image" :style="{ background: cardGradient(item) }">
              <span class="card-image-label">{{ item.brand }} {{ item.model }}</span>
            </div>
            <div class="score" :class="{ web: item.source_type === 'web' }">{{ item.source_type === 'web' ? 'WEB' : item.score }}</div>
            <h3>{{ item.brand }} {{ item.model }}</h3>
            <p v-if="item.source_type === 'web'">联网候选 · 参数待核验</p>
            <p v-else>{{ item.energy_type }} · {{ item.vehicle_type }} · {{ item.cltc_range }}km</p>
            <div class="tags" v-if="item.source_type === 'web'">
              <span>Web Search</span><span>待入库</span><span>需核验</span>
            </div>
            <div class="tags" v-else><span>{{ item.price_min / 10000 }}-{{ item.price_max / 10000 }}万</span><span>{{ item.adas_level }}</span><span>{{ item.seats }}座</span></div>
            <ul><li v-for="reason in item.reasons" :key="reason">{{ reason }}</li></ul>
            <a v-if="item.source_type === 'web' && item.source_url" class="source-link" :href="normalizeUrl(item.source_url)" target="_blank">查看搜索来源</a>
            <div class="feedback-actions">
              <button @click="submitFeedback(item, 'positive')">👍 推荐准确</button>
              <button @click="submitFeedback(item, 'negative')">👎 需要优化</button>
            </div>
          </div>
        </div>

        <div class="grid three decision-grid" v-if="topRecommendation">
          <div class="card decision-card">
            <h3>推荐决策面板</h3>
            <div class="decision-hero" :style="{ background: cardGradient(topRecommendation) }">
              <span>首推车型</span>
              <b>{{ topRecommendation.brand }} {{ topRecommendation.model }}</b>
              <small>综合匹配 {{ topRecommendation.score }} 分</small>
            </div>
            <div class="decision-metrics">
              <div v-for="item in decisionHighlights" :key="item.label"><span>{{ item.label }}</span><b>{{ item.value }}</b></div>
            </div>
          </div>
          <div class="card">
            <h3>风险核验清单</h3>
            <div class="risk-list">
              <div v-for="risk in riskChecklist" :key="risk"><span>!</span><p>{{ risk }}</p></div>
            </div>
          </div>
          <div class="card">
            <h3>下一步跟进动作</h3>
            <div class="action-list">
              <div v-for="(action, index) in actionItems" :key="action"><b>{{ index + 1 }}</b><p>{{ action }}</p></div>
            </div>
            <el-button class="mt" type="primary" plain @click="saveLead">保存当前推荐为线索</el-button>
          </div>
        </div>

        <div class="card explainability-card" v-if="explainability">
          <div class="card-title"><h3>Agent 可解释决策报告</h3><span>{{ explainability.pool_decision?.selected_pool }} 候选池</span></div>
          <p class="muted">{{ explainability.pool_decision?.reason }}</p>
          <div class="explain-grid">
            <div v-for="item in explainability.top_comparisons || []" :key="item.model" class="explain-item">
              <b>Top {{ item.rank }} · {{ item.model }} · {{ item.score }}分</b>
              <p>{{ item.best_for }}</p>
              <small>{{ item.why_not_others }}</small>
              <ul><li v-for="reason in item.why_selected" :key="reason">{{ reason }}</li></ul>
              <div class="risk-list mini"><div v-for="risk in item.cautions" :key="risk"><span>!</span><p>{{ risk }}</p></div></div>
            </div>
          </div>
          <div class="grid two mt">
            <div><h4>谨慎/不推荐原因</h4><ul><li v-for="item in explainability.not_recommended || []" :key="item">{{ item }}</li></ul></div>
            <div><h4>试驾与跟进动作</h4><ul><li v-for="item in explainability.follow_up_actions || []" :key="item">{{ item }}</li></ul></div>
          </div>
        </div>

        <div class="card feedback-policy-card" v-if="feedbackPolicy?.applied_rules?.length">
          <div class="card-title"><h3>反馈策略解释面板</h3><span class="saved-pill">阶段 G1</span></div>
          <div class="policy-stability" v-if="feedbackPolicy?.policy?.stability">
            <div><span>车型样本阈值</span><b>{{ feedbackPolicy.policy.stability.min_model_samples }} 条</b></div>
            <div><span>候选池样本阈值</span><b>{{ feedbackPolicy.policy.stability.min_pool_samples }} 条</b></div>
            <div><span>时间衰减</span><b>{{ feedbackPolicy.policy.stability.uses_recency_decay ? '已启用' : '未启用' }}</b></div>
            <div><span>置信度</span><b>{{ feedbackPolicy.policy.stability.uses_confidence ? '已启用' : '未启用' }}</b></div>
          </div>
          <div class="policy-list">
            <div v-for="item in feedbackPolicy.applied_rules" :key="`${item.type}-${item.target}`">
              <b>{{ item.target }}</b>
              <span :class="item.delta < 0 ? 'warn-text' : 'ok-text'">{{ item.delta > 0 ? '+' : '' }}{{ item.delta }} 分</span>
              <small>{{ policyRuleText(item) }}</small>
            </div>
          </div>
        </div>

        <div class="grid two">
          <div class="card"><h3>Agent 推荐报告</h3><div class="answer" v-html="answerHtml"></div><div class="obsidian-saved" v-if="obsidianNote?.path"><b>已沉淀到 Obsidian Vault</b><p>{{ obsidianNote.title }}</p><small>{{ obsidianNote.path }}</small></div></div>
          <div class="card"><div class="card-title"><h3>Agent Trace 可视化</h3><span class="saved-pill">{{ agentTrace.length || 0 }} 步</span></div><div class="agent-trace-flow"><div v-for="(step, index) in agentTrace" :key="index"><span>{{ index + 1 }}</span><b>{{ step.agent }}</b><p>{{ step.observation }}</p></div><div v-if="!agentTrace.length" class="empty-case">生成推荐后展示画像解析、候选池选择、推荐排序、证据检索、风险核验、报告生成和 Obsidian 写入链路。</div></div></div>
        </div>

        <div class="card" v-if="evidenceRows.length">
          <h3>推荐证据与来源</h3>
          <el-table :data="evidenceRows" height="300">
            <el-table-column prop="rank" label="#" width="55" />
            <el-table-column prop="domain" label="来源类型" width="120" />
            <el-table-column prop="source" label="来源" width="190" />
            <el-table-column prop="score" label="相关度" width="90" />
            <el-table-column prop="content" label="证据片段" />
          </el-table>
        </div>
      </section>

      <section v-if="active === 'service'" class="section">
        <div class="chat-layout">
          <div class="card chat-card">
            <div class="chat-head">
              <div class="chat-robot">🤖</div>
              <div style="flex:1">
                <h3>智能客服</h3>
                <p>{{ serviceLoading ? '客服正在查询资料并输入中...' : '在线 · Agent + RAG + Web Search' }}</p>
              </div>
              <el-switch v-model="serviceUseWebSearch" active-text="联网" inactive-text="本地" />
            </div>
            <div class="chat-window">
              <div v-for="(msg, index) in chatMessages" :key="index" class="chat-message" :class="msg.role">
                <div class="bubble" v-html="toHtml(msg.content)"></div>
              </div>
              <div v-if="serviceLoading" class="chat-message assistant">
                <div class="bubble typing"><span></span><span></span><span></span> 正在思考输入中</div>
              </div>
            </div>
            <div class="chat-input">
              <el-input
                v-model="serviceQuestion"
                type="textarea"
                :rows="3"
                resize="none"
                placeholder="请输入客户问题，例如：没有家充应该买纯电、插混还是增程？"
                @keydown.ctrl.enter="askCustomerService"
              />
              <el-button type="primary" :loading="serviceLoading" @click="askCustomerService">发送</el-button>
            </div>
          </div>
          <div class="card">
            <h3>客服 Agent 调用链路</h3>
            <div class="timeline"><div v-for="(step, index) in serviceTrace" :key="index"><b>{{ step.agent }}</b><p>{{ step.observation }}</p></div></div>
          </div>
        </div>
        <div class="card">
          <h3>客服引用证据</h3>
          <el-table :data="sources" height="420">
            <el-table-column prop="rank" label="#" width="55" />
            <el-table-column prop="domain" label="来源类型" width="110" />
            <el-table-column prop="source" label="来源" width="180" />
            <el-table-column prop="score" label="分数" width="90" />
            <el-table-column prop="content" label="证据片段 / 搜索标题" />
          </el-table>
        </div>
      </section>

      <section v-if="active === 'compare'" class="section">
        <div class="card">
          <div class="card-title">
            <h3>竞品对比</h3>
            <div>
              <el-button @click="exportCompareCsv">导出 CSV</el-button>
              <el-button type="primary" @click="submitCompare">生成对比</el-button>
            </div>
          </div>
          <el-select v-model="compareModels" multiple filterable placeholder="选择 2-3 款车型">
            <el-option v-for="v in vehicles" :key="v.id" :label="`${v.brand} ${v.model}`" :value="`${v.brand} ${v.model}`" />
          </el-select>
        </div>
        <div class="grid two">
          <div class="card"><h3>竞品综合评分</h3><VChart class="chart" :option="compareScoreOption" autoresize /></div>
          <div class="card"><h3>价格 - 续航对比</h3><VChart class="chart" :option="compareScatterOption" autoresize /></div>
        </div>
        <div class="card">
          <h3>分项能力对比</h3>
          <VChart class="chart tall" :option="compareDimensionOption" autoresize />
        </div>
        <div class="card">
          <el-table :data="compareRows" height="460">
            <el-table-column prop="brand" label="品牌" width="90" />
            <el-table-column label="车型" width="150">
              <template #default="{ row }">
                <el-popover placement="right" :width="340" trigger="hover" @show="fetchCarImage(row)">
                  <template #reference>
                    <span class="model-link">{{ row.brand }} {{ row.model }}</span>
                  </template>
                  <div class="popover-car-card">
                    <img v-if="hoverCarImage" :src="hoverCarImage" class="popover-car-img" @error="hoverCarImage=''" />
                    <div v-if="!hoverCarImage" class="popover-car-placeholder" :style="{ background: cardGradient(row) }">
                      <span>{{ row.brand }} {{ row.model }}</span>
                    </div>
                    <div class="car-specs">
                      <span>{{ row.energy_type }}</span>
                      <span>{{ row.vehicle_type }}</span>
                      <span>{{ row.cltc_range }}km</span>
                      <span>{{ (row.price_min/10000)|0 }}-{{ (row.price_max/10000)|0 }}万</span>
                    </div>
                  </div>
                </el-popover>
              </template>
            </el-table-column>
            <el-table-column prop="score" label="推荐分" width="90" />
            <el-table-column prop="energy_type" label="能源" width="90" />
            <el-table-column prop="price_min" label="起售价" width="100" />
            <el-table-column prop="cltc_range" label="CLTC" width="90" />
            <el-table-column prop="highlights" label="亮点" />
            <el-table-column prop="weaknesses" label="短板" />
          </el-table>
        </div>
      </section>

      <section v-if="active === 'leads'" class="section">
        <div class="card">
          <div class="card-title"><h3>销售线索</h3><el-button type="primary" @click="saveLead">保存当前推荐为线索</el-button></div>
          <el-table :data="leads" height="520">
            <el-table-column prop="created_at" label="创建时间" width="170" />
            <el-table-column prop="name" label="客户" width="110" />
            <el-table-column prop="budget" label="预算" width="110" />
            <el-table-column prop="city" label="城市" width="110" />
            <el-table-column prop="concerns" label="关注点" />
            <el-table-column prop="intent_level" label="意向" width="100" />
            <el-table-column prop="recommended_models" label="推荐车型" />
            <el-table-column prop="next_action" label="下一步" />
          </el-table>
        </div>
      </section>

      <section v-if="active === 'knowledge'">
        <ObsidianKnowledge />
      </section>

      <section v-if="active === 'settings'" class="section">
        <div class="grid two">
          <div class="card">
            <h3>系统配置</h3>
            <div class="setting-row"><span>Base URL</span><b>{{ config?.base_url }}</b></div>
            <div class="setting-row"><span>Chat Model</span><b>{{ config?.chat_model }}</b></div>
            <div class="setting-row"><span>API Key</span><b>{{ config?.api_key_configured ? `已配置（${config?.api_key_masked || '已脱敏'}）` : '未配置' }}</b></div>
            <div class="setting-row"><span>LLM 连通性</span><b :class="config?.llm_available ? 'ok-text' : 'warn-text'">{{ config?.llm_available ? '可用' : '不可用/需端点权限' }}</b></div>
            <div class="setting-row"><span>内容生成模型</span><b>{{ config?.content_generation?.configured ? config?.content_generation?.model : '未配置' }}</b></div>
          </div>
          <div class="card danger">
            <h3>数据维护</h3>
            <p>可重建知识库索引或清空线索、推荐日志、会话等运行态数据。</p>
            <el-button @click="rebuildKnowledge">重建 RAG 索引</el-button>
            <el-button type="success" @click="seedDemo">补充演示数据</el-button>
            <el-button type="danger" @click="clearData">清空运行数据</el-button>
          </div>
        </div>
        <div class="card delivery-card">
          <div class="card-title"><h3>G7 版本说明与交付包</h3><el-button type="success" :loading="deliveryLoading" @click="runDeliveryPackage">生成交付包</el-button></div>
          <div class="release-status" :class="deliveryPackage.summary?.deliverable ? 'pass' : 'warn'">
            <div><span>交付结论</span><b>{{ deliveryPackage.summary?.deliverable ? '可交付' : '待生成/复核' }}</b></div>
            <div><span>交付评分</span><b>{{ deliveryPackage.summary?.delivery_score || 0 }}%</b></div>
            <div><span>验收评分</span><b>{{ deliveryPackage.summary?.acceptance_score || 0 }}%</b></div>
            <div><span>阶段通过</span><b>{{ deliveryPackage.summary?.passed_stage_count || 0 }}/{{ deliveryPackage.summary?.stage_count || 0 }}</b></div>
          </div>
          <div class="readiness-layout">
            <el-table :data="deliveryPackage.release_notes || []" height="260">
              <el-table-column prop="phase" label="阶段" width="90" />
              <el-table-column prop="title" label="交付内容" min-width="190" />
              <el-table-column prop="scope" label="说明" min-width="280" />
            </el-table>
            <div class="readiness-side">
              <h3>交付检查</h3>
              <div v-for="item in deliveryPackage.checklist || []" :key="item.name" class="release-action">
                <b>{{ item.passed ? '✅' : '❌' }} {{ item.name }}</b><span>{{ item.detail }}</span>
              </div>
              <p class="muted">报告：{{ deliveryPackage.summary?.markdown_report || '点击生成后落盘' }}</p>
            </div>
          </div>
        </div>
        <div class="card acceptance-card">
          <div class="card-title"><h3>G6 自动化验收报告</h3><el-button type="success" :loading="acceptanceLoading" @click="runPreReleaseCheck">一键发布前检查</el-button></div>
          <div class="release-status" :class="acceptanceReport.summary?.accepted ? 'pass' : 'warn'">
            <div><span>验收结论</span><b>{{ acceptanceReport.summary?.accepted ? '通过验收' : '待复核' }}</b></div>
            <div><span>综合评分</span><b>{{ acceptanceReport.summary?.acceptance_score || 0 }}%</b></div>
            <div><span>阶段通过</span><b>{{ acceptanceReport.summary?.passed_stage_count || 0 }}/{{ acceptanceReport.summary?.stage_count || 0 }}</b></div>
            <div><span>门禁状态</span><b>{{ acceptanceReport.summary?.release_gate_status || '待运行' }}</b></div>
          </div>
          <div class="readiness-layout">
            <el-table :data="acceptanceReport.stages || []" height="260">
              <el-table-column prop="stage" label="阶段" min-width="180" />
              <el-table-column label="状态" width="90"><template #default="{ row }"><el-tag :type="row.status === 'pass' ? 'success' : 'warning'">{{ row.status === 'pass' ? '通过' : '待复核' }}</el-tag></template></el-table-column>
              <el-table-column prop="pass_rate" label="通过率" width="100" />
              <el-table-column prop="file" label="报告文件" min-width="260" />
            </el-table>
            <div class="readiness-side">
              <h3>验收动作</h3>
              <div v-for="item in acceptanceReport.next_actions || []" :key="item" class="release-action">{{ item }}</div>
              <p class="muted">Markdown：{{ acceptanceReport.summary?.markdown_report || '点击一键检查后生成' }}</p>
            </div>
          </div>
        </div>
        <div class="card release-gate-card">
          <div class="card-title"><h3>Agent 发布门禁</h3><el-button type="primary" @click="refreshReleaseGate">运行门禁</el-button></div>
          <div class="release-status" :class="releaseGate.summary?.status || 'blocked'">
            <div><span>发布结论</span><b>{{ releaseGate.summary?.release_allowed ? '允许发布' : '暂缓发布' }}</b></div>
            <div><span>门禁评分</span><b>{{ releaseGate.summary?.gate_score || 0 }}%</b></div>
            <div><span>阻断项</span><b>{{ releaseGate.summary?.blocker_count || 0 }}</b></div>
            <div><span>警告项</span><b>{{ releaseGate.summary?.warning_count || 0 }}</b></div>
          </div>
          <div class="release-metrics">
            <div><span>工程就绪</span><b>{{ releaseGate.metrics?.readiness_score || 0 }}%</b></div>
            <div><span>Agent回归</span><b>{{ releaseGate.metrics?.agent_pass_rate || 0 }}%</b></div>
            <div><span>数据治理</span><b>{{ releaseGate.metrics?.governance_score || 0 }}</b></div>
            <div><span>真实样本</span><b>{{ releaseGate.metrics?.real_record_count || 0 }}</b></div>
            <div><span>反馈样本</span><b>{{ releaseGate.metrics?.feedback_total || 0 }}</b></div>
          </div>
          <div class="readiness-layout">
            <el-table :data="releaseGate.gate_items || []" height="300">
              <el-table-column prop="name" label="门禁项" min-width="170" />
              <el-table-column label="状态" width="90"><template #default="{ row }"><el-tag :type="row.passed ? 'success' : row.level === 'blocker' ? 'danger' : 'warning'">{{ row.passed ? '通过' : row.level === 'blocker' ? '阻断' : '警告' }}</el-tag></template></el-table-column>
              <el-table-column prop="actual" label="当前值" width="110" />
              <el-table-column prop="threshold" label="阈值" width="130" />
              <el-table-column prop="action" label="处理动作" min-width="240" />
            </el-table>
            <div class="readiness-side">
              <h3>下一步动作</h3>
              <div v-for="item in releaseGate.next_actions || []" :key="item" class="release-action">{{ item }}</div>
              <p class="muted">生成时间：{{ releaseGate.summary?.generated_at || '待运行' }}</p>
            </div>
          </div>
        </div>
        <div class="card readiness-card">
          <div class="card-title"><h3>工程化健康检查</h3><el-button type="primary" @click="refreshSystemReadiness">刷新检查</el-button></div>
          <div class="regression-kpis">
            <div><span>就绪评分</span><b>{{ readiness.summary?.readiness_score || 0 }}%</b></div>
            <div><span>检查项</span><b>{{ readiness.summary?.check_count || 0 }}</b></div>
            <div><span>通过项</span><b>{{ readiness.summary?.passed_count || 0 }}</b></div>
            <div><span>风险数</span><b>{{ readiness.risks?.length || 0 }}</b></div>
          </div>
          <div class="readiness-layout">
            <el-table :data="readiness.checks || []" height="320">
              <el-table-column prop="name" label="检查项" min-width="180" />
              <el-table-column label="状态" width="90"><template #default="{ row }"><el-tag :type="row.passed ? 'success' : 'warning'">{{ row.passed ? '通过' : '待处理' }}</el-tag></template></el-table-column>
              <el-table-column prop="detail" label="详情" min-width="260" />
            </el-table>
            <div class="readiness-side">
              <h3>关键文件状态</h3>
              <div v-for="item in readiness.files || []" :key="item.path" class="readiness-file">
                <b>{{ item.name }}</b><span>{{ item.exists ? '存在' : '缺失' }} · {{ item.path }}</span>
              </div>
              <h3>发布前风险</h3>
              <div v-for="item in readiness.risks || []" :key="item.title" class="readiness-risk">
                <b>{{ item.level }} · {{ item.title }}</b><span>{{ item.action }}</span>
              </div>
              <p v-if="!(readiness.risks || []).length" class="muted">当前未发现阻断发布风险。</p>
            </div>
          </div>
        </div>
      </section>
    </main>
    <div class="watermark">soldier_yhl</div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { animate } from 'animejs'
import { gsap } from 'gsap'
import { ElMessage, ElMessageBox } from 'element-plus'
import { API_BASE_URL, checkLlmConfig, clearRuntimeData, compare, createLead, createRecommendationFeedback, customerServiceChat, generateDeliveryPackage, getFusedCatalog, getHealth, getLeads, getOptimizationInsights, getRealWorldOverview, getRecommendationFeedbackReview, getRecommendationFeedbackSummary, getReleaseGate, getSummary, getSystemReadiness, getVehicles, previewProfile, publicConfig, rebuildRag, recommendAgent, recommendFused, recommendRealWorld, refreshRealWorldGovernance, runAcceptanceReport, runAgentRegressionEvaluation, runRecommendationEvaluation, seedDemoData } from './api/client'
import ObsidianKnowledge from './components/ObsidianKnowledge.vue'

const navs = [
  { key: 'recommend', label: 'Agent 工作台', icon: '01' },
  { key: 'feedback', label: '反馈复盘', icon: '02' },
  { key: 'optimization', label: '优化建议', icon: '03' },
  { key: 'evaluation', label: '质量评估', icon: '04' },
  { key: 'knowledge', label: 'Obsidian 知识库', icon: '05' },
  { key: 'dashboard', label: '销售总览', icon: '06' },
  { key: 'realdata', label: '真实数据', icon: '07' },
  { key: 'service', label: '智能客服', icon: '08' },
  { key: 'compare', label: '竞品对比', icon: '09' },
  { key: 'leads', label: '销售线索', icon: '10' },
  { key: 'settings', label: '系统设置', icon: '11' }
]

const subtitles: any = {
  dashboard: '辅助查看车型库、线索、推荐日志、预算分布、关注点和知识库状态。',
  evaluation: '辅助回放固定测试集，持续评估画像解析、Top 推荐命中和问题诊断。',
  feedback: '辅助沉淀人工好评/差评和原因，反向驱动 Agent 自我复盘。',
  optimization: '辅助把质量评估和人工反馈汇总成可执行优化任务，形成持续迭代路线。',
  realdata: '辅助查看 200+ 条真实新能源车型数据的数据质量、字段缺口和真实样本推荐评估结果。',
  recommend: '主入口：输入一句购车需求，Agent 自动解析画像、选择候选池、调用工具、生成报告、沉淀 Obsidian 并接收反馈。',
  service: '面向销售顾问和客户咨询的 Agent 智能客服，支持 Web Search、RAG 和合规检查。',
  compare: '围绕价格、续航、空间、智驾、补能和场景做竞品对比。',
  leads: '沉淀客户画像、推荐车型和后续跟进动作。',
  knowledge: '直接读取项目 Obsidian Vault，展示知识节点、双向链接和数据抓取补充结果。',
  settings: '查看模型配置、RAG 状态和运行数据维护。'
}

const initialActive = window.location.hash.replace('#', '')
const active = ref(navs.some(item => item.key === initialActive) ? initialActive : 'recommend')
const currentTitle = computed(() => navs.find(x => x.key === active.value)?.label || '')
const currentSubtitle = computed(() => subtitles[active.value] || '')
const summary = ref<any>(null)
const vehicles = ref<any[]>([])
const leads = ref<any[]>([])
const config = ref<any>(null)
const readiness = ref<any>({ summary: {}, checks: [], files: [], risks: [] })
const releaseGate = ref<any>({ summary: {}, metrics: {}, gate_items: [], blockers: [], warnings: [], next_actions: [] })
const acceptanceReport = ref<any>({ summary: {}, stages: [], next_actions: [] })
const acceptanceLoading = ref(false)
const deliveryPackage = ref<any>({ summary: {}, release_notes: [], key_files: [], checklist: [], usage_steps: [], next_actions: [] })
const deliveryLoading = ref(false)
const backendStatus = ref<any>({ status: 'checking', detail: '正在检测后端连接' })
const evaluationLoading = ref(false)
const evaluationSummary = ref<any>({ case_count: 0, pass_rate: 0, average_score: 0, warned: 0, failed: 0 })
const evaluationCases = ref<any[]>([])
const evaluationNote = ref<any>(null)
const agentRegression = ref<any>({ summary: {}, cases: [], obsidian_note: null })
const feedbackSummary = ref<any>({ total: 0, positive_rate: 0, positive: 0, negative: 0, reasons: [], recent: [], model_rows: [], pool_rows: [], scene_rows: [] })
const feedbackReview = ref<any>({ summary: {}, insights: [], obsidian_note: null })
const optimization = ref<any>({ summary: {}, items: [], evaluation: {}, feedback: {}, obsidian_note: null })
const optimizationLoading = ref(false)
const realWorld = ref<any>({ quality: {}, evaluation: {}, governance: { summary: {}, duplicates: [], missing_fields: [], anomalies: [], source_trust: {}, actions: [] }, stats: {}, samples: [], files: {} })
const realWorldLoading = ref(false)
const realWorldRecs = ref<any[]>([])
const realWorldRecLoading = ref(false)
const fusedCatalog = ref<any>({ summary: {}, vehicles: [] })
const fusedRecs = ref<any[]>([])
const fusedLoading = ref(false)
const loading = ref(false)
const serviceLoading = ref(false)
const query = ref('预算 25 万以内，三口之家，上海通勤每天 50 公里，有家充，关注续航、空间和智驾，推荐哪几款新能源 SUV？')
const useDeepSearch = ref(true)
const candidatePool = ref('auto')
const candidatePoolOptions = [
  { label: 'Agent自动', value: 'auto' },
  { label: '本地精选', value: 'local' },
  { label: '真实扩展', value: 'real' },
  { label: '融合池', value: 'fused' }
]
const serviceQuestion = ref('客户问：没有家充的家庭用户应该选择纯电、插混还是增程？请给出专业、合规、可执行的回答。')
const serviceUseWebSearch = ref(true)
const chatMessages = ref<any[]>([
  { role: 'assistant', content: '您好，我是智能客服。可以帮您解答车型选择、充电续航、智能驾驶、价格权益和竞品对比问题。' }
])
const profile = ref<any>({ budget_max: 250000, city: '', family_size: null, commute_km: null, has_home_charger: null, preferred_type: '', preferred_energy: '', concerns: [] })
const profilePreview = ref<any>(null)
const profileParsingLoading = ref(false)
const recommendations = ref<any[]>([])
const answerHtml = ref('<span class="muted">点击生成推荐后，系统会展示推荐报告。</span>')
const serviceAnswerHtml = ref('<span class="muted">客服回答会显示在这里。</span>')
const agentTrace = ref<any[]>([])
const serviceTrace = ref<any[]>([])
const skillTrace = ref<any[]>([])
const sources = ref<any[]>([])
const explainability = ref<any>(null)
const feedbackPolicy = ref<any>(null)
const obsidianNote = ref<any>(null)
const compareModels = ref<any[]>(['特斯拉 Model Y', '小鹏 G6', '比亚迪 宋L EV'])
const compareRows = ref<any[]>([])
const hoverCarImage = ref('')
const staticFallbackMode = false
const backendStatusText = computed(() => backendStatus.value.status === 'ok' ? '后端在线' : backendStatus.value.status === 'checking' ? '检测中' : '后端离线')
const backendStatusDetail = computed(() => backendStatus.value.detail || API_BASE_URL)
const selectedPoolLabel = computed(() => explainability.value?.pool_decision?.selected_pool || (candidatePool.value === 'auto' ? '待 Agent 自动选择' : candidatePool.value))
const agentWorkspaceStats = computed(() => [
  { label: '当前候选池', value: selectedPoolLabel.value, desc: explainability.value?.pool_decision?.reason || '生成推荐后展示后端 Agent 决策原因' },
  { label: '工具调用', value: `${agentTrace.value.length || 0} 步`, desc: agentTrace.value.length ? 'Trace 已返回，可查看完整编排链路' : '待生成推荐后展示工具调用' },
  { label: '推荐证据', value: `${evidenceRows.value.length || 0} 条`, desc: sources.value.length ? '包含 RAG / 规则 / 数据来源证据' : '默认展示本地规则证据' },
  { label: '长期记忆', value: obsidianNote.value?.path ? '已沉淀' : '待写入', desc: obsidianNote.value?.path || '推荐完成后写入 Obsidian Vault' },
  { label: '反馈策略', value: `${feedbackPolicy.value?.applied_rules?.length || 0} 条`, desc: feedbackPolicy.value?.applied_rules?.length ? '已按历史反馈调整推荐分' : '推荐后展示反馈加权/降权结果' },
])
const auxiliaryViews = [
  { key: 'feedback', title: '反馈复盘', desc: '查看候选池质量、场景风险和 Agent 自我复盘' },
  { key: 'optimization', title: '优化建议', desc: '把评估和反馈转成可执行优化任务' },
  { key: 'evaluation', title: '质量评估', desc: '回放固定用例验证推荐链路稳定性' },
  { key: 'knowledge', title: 'Obsidian 记忆', desc: '查看推荐案例、知识节点和图谱连接' },
]

const demoVehicles = [
  { id: 1, brand: '比亚迪', model: '宋PLUS DM-i', vehicle_type: 'SUV', energy_type: '插混', price_min: 129800, price_max: 169800, cltc_range: 1100, seats: 5, adas_level: 'L2', safety_score: 92, monthly_sales: 32000, highlights: '插混油耗低;空间实用;售后网点多', weaknesses: '高速后段动力一般;智能驾驶保守' },
  { id: 2, brand: '比亚迪', model: '宋L EV', vehicle_type: 'SUV', energy_type: '纯电', price_min: 189800, price_max: 249800, cltc_range: 662, seats: 5, adas_level: 'L2', safety_score: 91, monthly_sales: 12000, highlights: '纯电平台;外观运动;空间表现好', weaknesses: '品牌溢价感一般;后备箱容积中等' },
  { id: 3, brand: '特斯拉', model: 'Model Y', vehicle_type: 'SUV', energy_type: '纯电', price_min: 263900, price_max: 363900, cltc_range: 688, seats: 5, adas_level: 'L2', safety_score: 94, monthly_sales: 41000, highlights: '能耗控制优秀;补能网络成熟;保值率高', weaknesses: '内饰简约;价格波动' },
  { id: 4, brand: '小鹏', model: 'G6', vehicle_type: 'SUV', energy_type: '纯电', price_min: 199900, price_max: 276900, cltc_range: 755, seats: 5, adas_level: 'L2+', safety_score: 90, monthly_sales: 8500, highlights: '800V 快充;智驾能力强;性价比高', weaknesses: '品牌保值率需观察;后排舒适性一般' },
  { id: 5, brand: '理想', model: 'L6', vehicle_type: 'SUV', energy_type: '增程', price_min: 249800, price_max: 279800, cltc_range: 1390, seats: 5, adas_level: 'L2+', safety_score: 93, monthly_sales: 23000, highlights: '空间舒适;增程适合长途;座舱体验强', weaknesses: '纯电续航有限;车重较高' },
  { id: 6, brand: '问界', model: 'M7', vehicle_type: 'SUV', energy_type: '增程', price_min: 249800, price_max: 329800, cltc_range: 1300, seats: 5, adas_level: 'L2+', safety_score: 93, monthly_sales: 19000, highlights: '鸿蒙生态;舒适配置强;主动安全丰富', weaknesses: '车型较大;第三方口碑分化' }
]

function setActive(key: string) {
  active.value = key
  window.location.hash = key
  playSectionMotion()
}

function allowMotion() {
  return !window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function playShellMotion() {
  if (!allowMotion()) return
  nextTick(() => {
    gsap.fromTo('.hero', { autoAlpha: 0, y: -18, filter: 'blur(8px)' }, { autoAlpha: 1, y: 0, filter: 'blur(0px)', duration: .72, ease: 'power3.out' })
    gsap.fromTo('.sidebar .nav-item', { autoAlpha: 0, x: -16 }, { autoAlpha: 1, x: 0, duration: .48, stagger: .035, ease: 'power2.out' })
    animate('.logo', { scale: [1, 1.08, 1], rotate: ['0deg', '6deg', '0deg'], duration: 2800, loop: true, ease: 'inOutSine' })
    animate('.backend-status span', { scale: [1, 1.35, 1], opacity: [.45, 1, .45], duration: 1800, loop: true, ease: 'inOutSine' })
    playSectionMotion()
  })
}

function playSectionMotion() {
  if (!allowMotion()) return
  nextTick(() => {
    gsap.fromTo('.section .card, .section .kpi, .section .vehicle-card, .section .release-status > div', { autoAlpha: 0, y: 18, scale: .985 }, { autoAlpha: 1, y: 0, scale: 1, duration: .55, stagger: .035, ease: 'power3.out', overwrite: 'auto' })
    const kpiValues = document.querySelectorAll('.section .kpi strong')
    if (kpiValues.length) animate(kpiValues, { opacity: [0, 1], translateY: [8, 0], duration: 650, delay: (_: any, i: number) => i * 70, ease: 'outCubic' })
  })
}

async function fetchCarImage(row: any) {
  hoverCarImage.value = ''
  try {
    const resp = await fetch(`${API_BASE_URL}/car-image?brand=${encodeURIComponent(row.brand)}&model=${encodeURIComponent(row.model)}`)
    const data = await resp.json()
    hoverCarImage.value = data.image_url || ''
  } catch { hoverCarImage.value = '' }
}

function money(value: number) {
  if (!value) return '--'
  return `${Math.round(value / 10000)}万`
}

function toHtml(text: string) {
  return (text || '').replaceAll('\n', '<br/>')
}

function policyRuleText(item: any) {
  const direction = item.delta < 0 ? '降权' : '加权'
  const confidence = item.confidence ? `｜置信度 ${item.confidence}` : ''
  const latest = item.latest_feedback_at ? `｜最近反馈 ${item.latest_feedback_at}` : ''
  if (item.type === 'pool') return `${direction}｜样本 ${item.total || 0} 条｜正反馈率 ${item.positive_rate || 0}%${confidence}${latest}`
  return `${direction}｜正 ${item.positive || 0} / 负 ${item.negative || 0}｜样本 ${item.total || 0} 条${confidence}${latest}`
}

function normalizeUrl(url: string) {
  if (!url) return '#'
  if (url.startsWith('//')) return `https:${url}`
  return url
}

function cardGradient(item: any) {
  const map: Record<string, string> = {
    '小鹏':   'linear-gradient(135deg, #0d9488 0%, #0f766e 100%)',
    '理想':   'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
    '特斯拉': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
    '比亚迪': 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)',
    '问界':   'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
    '享界':   'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
    '尊界':   'linear-gradient(135deg, #d97706 0%, #b45309 100%)',
    '智界':   'linear-gradient(135deg, #059669 0%, #047857 100%)',
    '蔚来':   'linear-gradient(135deg, #0891b2 0%, #0e7490 100%)',
    '极氪':   'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
    '小米':   'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
    '腾势':   'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
    '阿维塔': 'linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)',
    '宝马':   'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
    '奔驰':   'linear-gradient(135deg, #8899aa 0%, #556b82 100%)',
    '奥迪':   'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
  }
  return map[item.brand] || 'linear-gradient(135deg, #0f766e 0%, #115e59 100%)'
}

const energyOption = computed(() => pieOption(summary.value?.energy_distribution || {}))
const budgetOption = computed(() => barOption(summary.value?.budget_distribution || {}, '#2878c7'))
const concernOption = computed(() => barOption(summary.value?.concern_distribution || {}, '#1f7a4d'))
const topRecommendation = computed(() => recommendations.value[0] || null)
const decisionHighlights = computed(() => {
  const top = topRecommendation.value
  if (!top) return []
  return [
    { label: '预算匹配', value: `${top.budget_score || 0}分` },
    { label: '补能适配', value: `${top.charging_score || 0}分` },
    { label: '空间表现', value: `${top.space_score || 0}分` },
    { label: '智驾能力', value: `${top.smart_score || 0}分` }
  ]
})
const riskChecklist = computed(() => {
  const top = topRecommendation.value
  const risks = [...(top?.cautions || [])]
  if (profile.value.has_home_charger === false) risks.unshift('确认居住地/公司 3km 内公共补能便利性')
  if ((profile.value.concerns || []).includes('智驾')) risks.push('试驾时核验辅助驾驶可用范围和接管提示')
  return risks.length ? risks.slice(0, 5) : ['核验官方实时价格、权益、质保和试驾体验']
})
const actionItems = computed(() => {
  const top = topRecommendation.value
  if (!top) return ['先补充预算、城市、通勤和充电条件', '生成推荐后再安排跟进']
  return [
    `优先邀约试驾 ${top.brand} ${top.model}，同步体验空间、座舱和底盘`,
    '确认客户充电条件、停车环境和长途频率',
    '准备 1-2 款备选车型，避免单一推荐导致流失',
    '把本次画像和推荐结论沉淀为销售线索'
  ]
})
const evidenceRows = computed(() => sources.value.length ? sources.value.slice(0, 6) : [
  { rank: 1, domain: '本地车型库', source: 'vehicle_database.csv', score: 1, content: '基于价格、能源类型、续航、空间、智驾和安全评分进行本地结构化推荐。' },
  { rank: 2, domain: '画像解析', source: '自然语言需求', score: profilePreview.value?.confidence || 0, content: profilePreview.value?.summary || '根据预算、家庭人数、通勤、家充和关注点生成推荐画像。' },
  { rank: 3, domain: '风险提示', source: '推荐规则', score: 0.86, content: '价格、权益、续航和辅助驾驶可用范围需以官方实时信息和试驾体验为准。' }
])
const evaluationIssues = computed(() => evaluationCases.value.flatMap((item: any) => (item.failed_checks || []).map((check: any, index: number) => ({ ...check, caseName: item.name, key: `${item.id}-${index}` }))))
const evaluationOption = computed(() => ({
  tooltip: {},
  grid: { left: 56, right: 20, top: 24, bottom: 70 },
  xAxis: { type: 'category', data: evaluationCases.value.map((item: any) => item.name), axisLabel: { color: '#334e68', rotate: 25 } },
  yAxis: { type: 'value', max: 100, splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: [{ type: 'bar', data: evaluationCases.value.map((item: any) => item.score), itemStyle: { color: (p: any) => evaluationCases.value[p.dataIndex]?.status === 'pass' ? '#0f766e' : evaluationCases.value[p.dataIndex]?.status === 'warn' ? '#f59e0b' : '#ef4444', borderRadius: [4, 4, 0, 0] }, barWidth: '48%' }]
}))
const feedbackOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { bottom: 0, textStyle: { color: '#556b82' } },
  grid: { left: 52, right: 18, top: 24, bottom: 64 },
  xAxis: { type: 'category', data: (feedbackSummary.value.model_rows || []).map((item: any) => item.model_name), axisLabel: { color: '#334e68', rotate: 20 } },
  yAxis: { type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: [
    { name: '正反馈', type: 'bar', stack: 'feedback', data: (feedbackSummary.value.model_rows || []).map((item: any) => item.positive), itemStyle: { color: '#0f766e', borderRadius: [4, 4, 0, 0] } },
    { name: '负反馈', type: 'bar', stack: 'feedback', data: (feedbackSummary.value.model_rows || []).map((item: any) => item.negative), itemStyle: { color: '#ef4444' } }
  ]
}))
const realBrandOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  grid: { left: 92, right: 20, top: 20, bottom: 24 },
  xAxis: { type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  yAxis: { type: 'category', data: (realWorld.value.stats?.brand_distribution || []).map((item: any) => item[0]).reverse(), axisLabel: { color: '#334e68' } },
  series: [{ type: 'bar', data: (realWorld.value.stats?.brand_distribution || []).map((item: any) => item[1]).reverse(), itemStyle: { color: '#0f766e', borderRadius: [0, 4, 4, 0] } }]
}))
const realTypeOption = computed(() => pieOption(Object.fromEntries(realWorld.value.stats?.vehicle_type_distribution || [])))
const realMissingRows = computed(() => Object.entries(realWorld.value.quality?.missing_counts || {}).map(([field, count]) => ({ field, count })).sort((a: any, b: any) => b.count - a.count).slice(0, 8))
const realEstimatedRows = computed(() => Object.entries(realWorld.value.enrichment?.estimated_field_counts || {}).map(([field, count]) => ({ field, count })).sort((a: any, b: any) => b.count - a.count).slice(0, 8))
const realEvaluationRows = computed(() => realWorld.value.evaluation?.cases || [])
const hotModelOption = computed(() => ({
  tooltip: {},
  grid: { left: 90, right: 24, top: 24, bottom: 24 },
  xAxis: { type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } }, axisLabel: { color: '#8899aa' } },
  yAxis: { type: 'category', data: (summary.value?.hot_models || []).map((x: any) => `${x.brand} ${x.model}`).reverse(), axisLabel: { color: '#334e68' } },
  series: [{ type: 'bar', data: (summary.value?.hot_models || []).map((x: any) => x.monthly_sales).reverse(), itemStyle: { color: '#0f766e', borderRadius: [0, 4, 4, 0], shadowBlur: 6, shadowColor: 'rgba(15,118,110,.2)', shadowOffsetX: 2 }, barWidth: '55%' }]
}))
const scatterOption = computed(() => ({
  tooltip: { formatter: (p: any) => `${p.data[2]}<br/>价格：${p.data[0]}万<br/>续航：${p.data[1]}km` },
  grid: { left: 48, right: 20, top: 24, bottom: 36 },
  xAxis: { name: '价格万', type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  yAxis: { name: 'CLTC km', type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: [{ type: 'scatter', symbolSize: 16, data: vehicles.value.map(v => [Math.round(((v.price_min + v.price_max) / 2) / 10000), v.cltc_range, `${v.brand} ${v.model}`]), itemStyle: { color: '#0f766e', shadowBlur: 10, shadowColor: 'rgba(15,118,110,.3)' } }]
}))
const radarOption = computed(() => {
  const top = recommendations.value[0] || {}
  return {
    tooltip: {},
    radar: { indicator: [
      { name: '预算', max: 100 }, { name: '续航', max: 100 }, { name: '空间', max: 100 },
      { name: '补能', max: 100 }, { name: '智驾', max: 100 }, { name: '安全', max: 100 }
    ], center: ['50%', '52%'], radius: '62%', splitArea: { areaStyle: { color: ['#fafbfc', '#f4f6f8', '#fafbfc', '#f4f6f8', '#fafbfc'] } } },
    series: [{ type: 'radar', data: [{ value: [top.budget_score || 0, top.range_score || 0, top.space_score || 0, top.charging_score || 0, top.smart_score || 0, top.safety_score || 0], name: top.model || '待推荐', areaStyle: { color: 'rgba(15,118,110,.2)' }, lineStyle: { color: '#0f766e', width: 2 }, itemStyle: { color: '#0f766e' }, symbol: 'circle', symbolSize: 6 }] }]
  }
})
const compareScoreOption = computed(() => ({
  tooltip: {},
  grid: { left: 56, right: 20, top: 24, bottom: 42 },
  xAxis: { type: 'category', data: compareRows.value.map(x => `${x.brand} ${x.model}`), axisLabel: { color: '#334e68' } },
  yAxis: { type: 'value', max: 100, splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: [{ type: 'bar', data: compareRows.value.map(x => x.score || 0), itemStyle: { color: '#0f766e', borderRadius: [4, 4, 0, 0], shadowBlur: 8, shadowColor: 'rgba(15,118,110,.2)', shadowOffsetY: 3 }, barWidth: '50%' }]
}))
const compareScatterOption = computed(() => ({
  tooltip: { formatter: (p: any) => `${p.data[2]}<br/>起售价：${p.data[0]}万<br/>CLTC：${p.data[1]}km` },
  grid: { left: 56, right: 20, top: 24, bottom: 42 },
  xAxis: { name: '起售价万', type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  yAxis: { name: 'CLTC km', type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: [{ type: 'scatter', symbolSize: 18, data: compareRows.value.map(x => [Math.round((x.price_min || 0) / 10000), x.cltc_range || 0, `${x.brand} ${x.model}`]), itemStyle: { color: '#0f766e', shadowBlur: 10, shadowColor: 'rgba(15,118,110,.3)' } }]
}))
const compareDimensionOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { bottom: 0, textStyle: { color: '#556b82' } },
  grid: { left: 48, right: 20, top: 32, bottom: 64 },
  xAxis: { type: 'category', data: ['预算', '续航', '空间', '补能', '智驾', '安全'], axisLabel: { color: '#334e68' } },
  yAxis: { type: 'value', max: 100, splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } } },
  series: compareRows.value.map((x, index) => ({
    name: `${x.brand} ${x.model}`,
    type: 'bar',
    data: [x.budget_score, x.range_score, x.space_score, x.charging_score, x.smart_score, x.safety_score],
    itemStyle: { borderRadius: [4, 4, 0, 0], shadowBlur: 4, shadowColor: 'rgba(0,0,0,.08)', shadowOffsetY: 2 },
    barGap: '20%',
    barCategoryGap: '30%'
  }))
}))

function pieOption(data: any) {
  const colors = ['#0f766e', '#39d98a', '#0ea5e9', '#8b5cf6', '#f59e0b', '#14b8a6', '#6366f1', '#ec4899']
  return {
    color: colors,
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: '#556b82' } },
    series: [{
      type: 'pie', radius: ['48%', '74%'],
      data: Object.entries(data).map(([name, value]) => ({ name, value })),
      itemStyle: { borderRadius: 4, borderColor: '#fff', borderWidth: 2,
        shadowBlur: 12, shadowColor: 'rgba(0,0,0,.08)', shadowOffsetY: 4 }
    }]
  }
}

function barOption(data: any, color: string) {
  return {
    tooltip: {},
    grid: { left: 52, right: 20, top: 24, bottom: 36 },
    xAxis: { type: 'category', data: Object.keys(data), axisLine: { lineStyle: { color: '#dce3e8' } }, axisLabel: { color: '#556b82' } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#eef1f5', type: 'dashed' } }, axisLabel: { color: '#8899aa' } },
    series: [{
      type: 'bar', data: Object.values(data),
      itemStyle: {
        color, borderRadius: [4, 4, 0, 0],
        shadowBlur: 8, shadowColor: color + '40', shadowOffsetY: 3
      },
      barWidth: '55%'
    }]
  }
}

function countBy(rows: any[], key: string) {
  return rows.reduce((acc: any, row: any) => {
    acc[row[key]] = (acc[row[key]] || 0) + 1
    return acc
  }, {})
}

function buildFallbackSummary(rows: any[]) {
  return {
    vehicle_count: rows.length,
    recommendation_count: recommendations.value.length || 12,
    avg_budget: profile.value.budget_max || 250000,
    rag_stats: { chunks: 36 },
    energy_distribution: countBy(rows, 'energy_type'),
    budget_distribution: { '15万内': 1, '15-25万': 3, '25-35万': 2 },
    concern_distribution: { 续航: 8, 空间: 7, 智驾: 6, 补能: 5, 安全: 5 },
    hot_models: rows.slice().sort((a, b) => b.monthly_sales - a.monthly_sales).slice(0, 6)
  }
}

function fallbackEvaluation() {
  const cases = [
    { id: 'family-no-home-charger', name: '三口之家无家充 SUV', status: 'pass', score: 100, top_models: ['比亚迪 宋PLUS DM-i', '理想 L6', '问界 M7'], failed_checks: [], diagnosis: '推荐链路符合当前测试预期' },
    { id: 'home-charger-pure-ev', name: '有家充纯电通勤', status: 'pass', score: 100, top_models: ['特斯拉 Model Y', '小鹏 G6', '比亚迪 宋L EV'], failed_checks: [], diagnosis: '推荐链路符合当前测试预期' },
    { id: 'explicit-compare', name: '点名车型对比', status: 'pass', score: 100, top_models: ['小鹏 G6', '特斯拉 Model Y'], failed_checks: [], diagnosis: '推荐链路符合当前测试预期' },
    { id: 'mpv-family', name: '多人家庭 MPV', status: 'pass', score: 88.9, top_models: ['极氪 009', '理想 L7', '理想 L6'], failed_checks: [], diagnosis: '推荐链路符合当前测试预期' },
    { id: 'social-luxury', name: '社交形象与豪华感', status: 'pass', score: 100, top_models: ['奔驰 E300L', '宝马 530Li', '享界 S9'], failed_checks: [], diagnosis: '推荐链路符合当前测试预期' }
  ]
  return { summary: { case_count: 5, passed: 5, warned: 0, failed: 0, pass_rate: 100, average_score: 97.8 }, cases, obsidian_note: { title: '静态演示评估报告', path: '后端恢复后写入 Obsidian Vault' } }
}

function fallbackAgentRegression() {
  const trace = ['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'ObsidianCaseWriterTool']
  const cases = [
    { id: 'agent-family-local', name: 'Agent家庭通勤本地池', status: 'pass', score: 100, selected_pool: 'local', top_models: ['理想 L6', '问界 M7'], trace_agents: trace, feedback_policy_rules: [], failed_checks: [], diagnosis: 'Agent端到端链路稳定' },
    { id: 'agent-no-home-fused', name: 'Agent无家充复杂场景', status: 'pass', score: 100, selected_pool: 'fused', top_models: ['理想 L6', '问界 M7'], trace_agents: trace, feedback_policy_rules: [{ target: 'fused' }], failed_checks: [], diagnosis: 'Agent端到端链路稳定' }
  ]
  return { summary: { case_count: cases.length, passed: cases.length, warned: 0, failed: 0, pass_rate: 100, average_score: 100 }, cases, obsidian_note: { title: '静态演示Agent回归报告', path: '后端恢复后写入 Obsidian Vault' } }
}

function fallbackFeedback() {
  return {
    total: 4,
    positive: 3,
    negative: 1,
    neutral: 0,
    positive_rate: 75,
    model_rows: [
      { model_name: '理想 L6', total: 2, positive: 2, negative: 0 },
      { model_name: '小鹏 G6', total: 1, positive: 1, negative: 0 },
      { model_name: '特斯拉 Model Y', total: 1, positive: 0, negative: 1 },
    ],
    pool_rows: [
      { candidate_pool: 'local', total: 2, positive: 2, negative: 0, positive_rate: 100 },
      { candidate_pool: 'fused', total: 1, positive: 1, negative: 0, positive_rate: 100 },
      { candidate_pool: 'real', total: 1, positive: 0, negative: 1, positive_rate: 0 },
    ],
    scene_rows: [
      { scenario: '预算冲突', total: 1, positive: 0, negative: 1, negative_rate: 100 },
      { scenario: '家庭SUV', total: 2, positive: 2, negative: 0, negative_rate: 0 },
    ],
    reasons: [{ reason: '推荐符合家庭场景', count: 2 }, { reason: '价格超预算，需要优化', count: 1 }],
    recent: [
      { created_at: '2026-06-22 13:45:00', model_name: '理想 L6', rating: 'positive', reason: '空间和补能都匹配' },
      { created_at: '2026-06-22 13:40:00', model_name: '特斯拉 Model Y', rating: 'negative', reason: '预算压力偏高' },
    ]
  }
}

function fallbackFeedbackReview() {
  return {
    summary: { feedback_total: 4, positive_rate: 75, pool_count: 3, risky_scene_count: 1, negative_model_count: 1, insight_count: 3 },
    insights: [
      { title: '候选池质量最高：local', evidence: '2 条反馈，正反馈率 100%。', action: '家庭 SUV 场景优先保留本地精选策略。' },
      { title: '候选池需复盘：real', evidence: '1 条反馈，正反馈率 0%。', action: '真实扩展数据进入中国销售场景前需要强化价格和地域核验。' },
      { title: '高风险场景：预算冲突', evidence: '1 条反馈，负反馈率 100%。', action: '推荐报告提前提示超预算风险，并给出低预算备选。' },
    ],
    obsidian_note: { title: '静态演示反馈复盘', path: '后端恢复后写入 Obsidian Vault' }
  }
}

function fallbackOptimization() {
  return {
    summary: { item_count: 3, p1_count: 1, feedback_total: 4, evaluation_pass_rate: 100 },
    items: [
      { title: '复盘负反馈车型：特斯拉 Model Y', priority: 'P1', source: '人工反馈', evidence: '存在预算压力偏高的负反馈。', action: '在无明确品牌偏好时降低超预算车型排序，并强化价格风险提示。' },
      { title: '扩大人工反馈样本量', priority: 'P2', source: '反馈闭环', evidence: '当前反馈样本不足 10 条。', action: '推荐后引导销售顾问持续点击准确/需优化并补充原因。' },
      { title: '补充更难的真实客户用例', priority: 'P3', source: '质量评估', evidence: '固定评估集已全部通过。', action: '加入预算冲突、品牌偏好冲突和无家充纯电偏好等边界用例。' },
    ],
    obsidian_note: { title: '静态演示优化建议', path: '后端恢复后写入 Obsidian Vault' }
  }
}

function fallbackReadiness() {
  const checks = [
    { name: '后端数据目录', passed: true, detail: 'data' },
    { name: '车型库可读取', passed: true, detail: `${demoVehicles.length} 条本地演示车型` },
    { name: 'RAG索引可用', passed: true, detail: '36 chunks（演示）' },
    { name: 'LLM密钥配置', passed: false, detail: '后端不可用，无法确认真实配置' },
    { name: '前端构建入口', passed: true, detail: 'frontend/dist/index.html' },
  ]
  return {
    summary: { generated_at: new Date().toLocaleString('zh-CN', { hour12: false }), check_count: checks.length, passed_count: 4, readiness_score: 80, vehicle_count: demoVehicles.length, rag_chunks: 36 },
    checks,
    files: [
      { name: '车型主库', path: 'data/vehicle_database.csv', exists: true, passed: true },
      { name: '真实数据候选库', path: 'data/real_world/real_ev_specs.csv', exists: true, passed: true },
      { name: '前端构建入口', path: 'frontend/dist/index.html', exists: true, passed: true },
    ],
    risks: [{ level: 'P1', title: '后端连接失败', action: '恢复 FastAPI 服务后重新刷新工程化健康检查' }]
  }
}

function fallbackDeliveryPackage() {
  const releaseNotes = [
    { phase: 'A-E', title: '基础能力', scope: '推荐工作台、RAG、Obsidian、真实数据与 Agent 编排' },
    { phase: 'F-G2', title: 'Agent策略闭环', scope: '反馈策略、稳定性、端到端回归' },
    { phase: 'G3-G6', title: '治理与发布', scope: '真实数据治理、健康检查、发布门禁、验收报告' },
  ]
  return {
    summary: { generated_at: new Date().toLocaleString('zh-CN', { hour12: false }), deliverable: false, delivery_score: 80, acceptance_score: acceptanceReport.value.summary?.acceptance_score || 92.5, stage_count: 6, passed_stage_count: 6, markdown_report: '后端恢复后生成' },
    release_notes: releaseNotes,
    key_files: [],
    checklist: [{ name: '后端交付接口', passed: false, detail: '后端不可用，需恢复后生成真实交付包' }],
    usage_steps: [],
    next_actions: ['恢复后端连接后重新生成交付包']
  }
}

function fallbackAcceptanceReport() {
  const stages = [
    { stage: 'F 反馈策略闭环', status: 'pass', pass_rate: 100, file: 'data/real_world/feedback_policy_evaluation.json' },
    { stage: 'G1 反馈策略稳定性', status: 'pass', pass_rate: 100, file: 'data/real_world/feedback_policy_stability_evaluation.json' },
    { stage: 'G2 Agent端到端回归', status: 'pass', pass_rate: 100, file: 'data/real_world/agent_regression_evaluation.json' },
    { stage: 'G3 真实数据治理', status: 'pass', pass_rate: 100, file: 'data/real_world/real_data_governance_evaluation.json' },
    { stage: 'G4 工程化健康检查', status: 'pass', pass_rate: 100, file: 'data/real_world/system_readiness_evaluation.json' },
    { stage: 'G5 发布门禁', status: 'pass', pass_rate: 100, file: 'data/real_world/release_gate_evaluation.json' },
  ]
  return {
    summary: { generated_at: new Date().toLocaleString('zh-CN', { hour12: false }), accepted: false, acceptance_score: 90, stage_count: stages.length, passed_stage_count: stages.length, release_gate_status: 'offline', markdown_report: '后端恢复后生成' },
    stages,
    next_actions: ['恢复后端连接后运行真实一键发布前检查']
  }
}

function fallbackReleaseGate() {
  const gateItems = [
    { name: '工程化就绪评分', passed: false, actual: 80, threshold: '>= 95', level: 'blocker', action: '恢复后端健康检查接口后重新运行门禁' },
    { name: 'Agent端到端回归', passed: true, actual: 100, threshold: '= 100', level: 'blocker', action: '保持回归用例持续通过' },
    { name: '真实数据治理评分', passed: true, actual: 93.8, threshold: '>= 90', level: 'blocker', action: '继续监控重复和异常数据' },
    { name: '真实数据样本量', passed: true, actual: 227, threshold: '>= 200', level: 'blocker', action: '继续补充真实车型数据' },
  ]
  return {
    summary: { generated_at: new Date().toLocaleString('zh-CN', { hour12: false }), status: 'blocked', release_allowed: false, gate_score: 75, passed_count: 3, gate_count: gateItems.length, blocker_count: 1, warning_count: 0 },
    metrics: { readiness_score: 80, agent_pass_rate: 100, governance_score: 93.8, real_record_count: 227, feedback_total: feedbackSummary.value.total || 0, feedback_positive_rate: feedbackSummary.value.positive_rate || 0 },
    gate_items: gateItems,
    blockers: gateItems.filter(item => !item.passed),
    warnings: [],
    next_actions: ['恢复 FastAPI 后端连接后重新运行真实发布门禁']
  }
}

function fallbackRealWorld() {
  return {
    quality: { source: 'open-ev-data/open-ev-data-dataset + OSkrk/Electric-vehicles-EV-Database', record_count: 227, unique_brand_count: 30, unique_model_count: 92, year_range: [2010, 2025], missing_counts: { trim: 67, vehicle_type_raw: 94, battery_kwh: 35, range_km: 2 } },
    enrichment: { avg_data_quality_score: 97, estimated_field_counts: { price: 227, seats: 227, wheelbase: 227, trunk: 227, safety: 227, adas: 227, battery_kwh: 35, range_km: 2 }, price_band_distribution: [['35-60万', 150], ['20-35万', 46], ['20万内', 25], ['60万以上', 6]] },
    evaluation: { case_count: 4, passed_count: 4, pass_rate: 100, cases: [] },
    governance: { summary: { quality_score: 88.5, duplicate_group_count: 12, anomaly_record_count: 4, trusted_source_rate: 96.5, missing_field_count: 198 }, duplicates: [{ brand: 'Audi', model: 'A6 e-tron', model_year: '2025', count: 2 }], anomalies: [{ brand: 'Demo', model: 'RangeX', model_year: '2025', issues: [{ message: '续航 1050km 异常' }] }], actions: [{ priority: 'P1', title: '合并重复车型版本', evidence: '发现重复车型 key', action: '保留来源可信度最高记录' }], source_trust: { levels: { high: 180, medium: 45, low: 2 } } },
    stats: { record_count: 227, avg_range_km: 420, max_range_km: 630, avg_battery_kwh: 67.8, avg_dc_charge_kw: 132.4, brand_distribution: [['Audi', 53], ['BMW', 45], ['Volkswagen', 30], ['Hyundai', 26]], vehicle_type_distribution: [['轿车', 174], ['SUV', 43], ['MPV', 7], ['微型车', 3]], year_distribution: [['2025', 74], ['2024', 54], ['2023', 32]] },
    samples: [
      { brand: 'BMW', model: 'iX', model_year: 2025, vehicle_type: 'SUV', range_km: 630, battery_kwh: 105.2, dc_charge_kw: 195, source_url: 'https://github.com/open-ev-data/open-ev-data-dataset' },
      { brand: 'Audi', model: 'Q4 e-tron', model_year: 2025, vehicle_type: '轿车', range_km: 341, battery_kwh: 48, dc_charge_kw: 125, source_url: 'https://github.com/open-ev-data/open-ev-data-dataset' },
    ],
    files: { normalized_csv: 'data/real_world/real_ev_specs.csv', recommender_csv: 'data/real_world/real_ev_specs_for_recommender.csv' }
  }
}

function fallbackFused() {
  return { summary: { total: 229, local_count: 38, real_count: 191, dedup_skipped: 29, brand_count: 46 }, vehicles: [] }
}

function localProfilePreview() {
  const text = query.value
  const parsed = { ...profile.value }
  const budget = text.match(/(\d+(?:\.\d+)?)\s*(万|w|W)/)
  if (budget) parsed.budget_max = Math.round(Number(budget[1]) * 10000)
  const km = text.match(/通勤.*?(\d+(?:\.\d+)?)\s*(公里|km|KM)/)
  if (km) parsed.commute_km = Number(km[1])
  if (text.includes('三口')) parsed.family_size = 3
  if (text.includes('二胎')) parsed.family_size = 4
  if (text.includes('上海')) parsed.city = '上海'
  else if (text.includes('北京')) parsed.city = '北京'
  if (['没有家充', '无家充', '不能装充电桩'].some(word => text.includes(word))) parsed.has_home_charger = false
  else if (['家充', '充电桩', '固定车位'].some(word => text.includes(word))) parsed.has_home_charger = true
  const concerns = ['续航', '空间', '智驾', '安全', '补能', '性价比'].filter(word => text.includes(word) || (word === '补能' && text.includes('充电')))
  parsed.concerns = Array.from(new Set([...(parsed.concerns || []), ...concerns]))
  const fields = [
    ['budget_max', '预算上限', parsed.budget_max ? `${Math.round(parsed.budget_max / 10000)}万` : '未识别'],
    ['city', '用车城市', parsed.city || '未识别'],
    ['family_size', '家庭人数', parsed.family_size ? `${parsed.family_size}人` : '未识别'],
    ['commute_km', '日常通勤', parsed.commute_km ? `${parsed.commute_km}公里/天` : '未识别'],
    ['has_home_charger', '家充条件', parsed.has_home_charger === true ? '有家充/固定车位' : parsed.has_home_charger === false ? '无家充/不确定可安装' : '未识别'],
    ['concerns', '核心关注点', parsed.concerns?.length ? parsed.concerns.join('、') : '未识别']
  ].map(([field, label, display]) => ({ field, label, display, detected: display !== '未识别', source: display !== '未识别' ? '前端本地解析' : '待补充' }))
  const detected = fields.filter((item: any) => item.detected).length
  return {
    profile: parsed,
    fields,
    confidence: Math.round(detected / fields.length * 100),
    summary: `已识别 ${detected}/${fields.length} 个关键画像字段`,
    missing_fields: fields.filter((item: any) => !item.detected).map((item: any) => item.label),
    insights: ['当前后端暂不可用，已启用前端本地画像解析兜底。', '你仍然可以查看推荐卡片、雷达图和基础解释，后端恢复后会自动使用完整 Agent 链路。']
  }
}

function localRecommendations() {
  const parsed = profilePreview.value?.profile || localProfilePreview().profile
  const budgetMax = parsed.budget_max || 250000
  return demoVehicles.map((item: any) => {
    let score = 78
    if (item.price_min <= budgetMax) score += 8
    if (parsed.preferred_type && item.vehicle_type === parsed.preferred_type) score += 5
    if (parsed.has_home_charger === false && ['插混', '增程'].includes(item.energy_type)) score += 7
    if ((parsed.concerns || []).includes('智驾') && item.adas_level.includes('+')) score += 4
    return {
      ...item,
      score: Math.min(score, 98),
      budget_score: item.price_min <= budgetMax ? 92 : 70,
      range_score: Math.min(Math.round(item.cltc_range / 8), 100),
      space_score: item.vehicle_type === 'SUV' ? 90 : 78,
      charging_score: parsed.has_home_charger === false && ['插混', '增程'].includes(item.energy_type) ? 94 : 82,
      smart_score: item.adas_level.includes('+') ? 92 : 80,
      safety_score: item.safety_score,
      reasons: [`价格区间 ${Math.round(item.price_min / 10000)}-${Math.round(item.price_max / 10000)} 万`, item.highlights, parsed.has_home_charger === false && ['插混', '增程'].includes(item.energy_type) ? '无家充场景补能更稳妥' : '适合作为候选车型进一步试驾确认'],
      cautions: [item.weaknesses]
    }
  }).sort((a, b) => b.score - a.score).slice(0, 5)
}

function applyLocalRecommend(reason = '后端接口暂不可用') {
  profilePreview.value = localProfilePreview()
  profile.value = { ...profile.value, ...profilePreview.value.profile }
  recommendations.value = localRecommendations()
  sources.value = []
  explainability.value = null
  const top = recommendations.value[0]
  agentTrace.value.push({ agent: 'LocalFallback', observation: `${reason}，已启用前端本地演示推荐兜底` })
  answerHtml.value = toHtml(`当前后端服务暂不可用，页面已切换为前端本地兜底模式。\n\n推荐优先关注 ${top.brand} ${top.model}。推荐理由：${top.reasons.join('；')}。\n\n后端恢复后，点击“生成推荐”会自动使用完整的 Agent + RAG + Obsidian 沉淀链路。`)
  obsidianNote.value = { title: '后端恢复后自动沉淀到 Obsidian Vault', path: '本地兜底模式暂不写入' }
}

async function runDeliveryPackage(showMessage = true) {
  deliveryLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用交付包兜底')
    deliveryPackage.value = await generateDeliveryPackage()
    if (showMessage) ElMessage.success(deliveryPackage.value.summary?.deliverable ? '交付包已生成：可交付' : '交付包已生成，仍需复核')
  } catch {
    deliveryPackage.value = fallbackDeliveryPackage()
    if (showMessage) ElMessage.warning('后端交付包接口不可用，已展示本地交付样例')
  } finally {
    deliveryLoading.value = false
  }
}

async function runPreReleaseCheck(showMessage = true) {
  acceptanceLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用验收兜底')
    acceptanceReport.value = await runAcceptanceReport()
    releaseGate.value = acceptanceReport.value.release_gate || releaseGate.value
    if (showMessage) ElMessage.success(acceptanceReport.value.summary?.accepted ? '自动化验收通过' : '验收报告已生成，仍需复核')
  } catch {
    acceptanceReport.value = fallbackAcceptanceReport()
    if (showMessage) ElMessage.warning('后端验收接口不可用，已展示本地验收样例')
  } finally {
    acceptanceLoading.value = false
  }
}

async function refreshReleaseGate(showMessage = true) {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用发布门禁兜底')
    releaseGate.value = await getReleaseGate()
    if (showMessage) ElMessage.success(releaseGate.value.summary?.release_allowed ? '发布门禁通过：允许进入验收' : '发布门禁完成：存在阻断项')
  } catch {
    releaseGate.value = fallbackReleaseGate()
    if (showMessage) ElMessage.warning('后端发布门禁接口不可用，已展示本地兜底门禁')
  }
}

async function refreshSystemReadiness(showMessage = true) {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用工程化兜底')
    readiness.value = await getSystemReadiness()
    if (showMessage) ElMessage.success(`工程化健康检查完成：${readiness.value.summary?.readiness_score || 0}%`)
  } catch {
    readiness.value = fallbackReadiness()
    if (showMessage) ElMessage.warning('后端健康检查接口不可用，已展示本地兜底检查')
  }
}

async function refresh() {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地兜底')
    const healthResp = await getHealth()
    backendStatus.value = { status: 'ok', detail: `车型 ${healthResp.vehicle_count} · RAG ${healthResp.rag_chunks} · uptime ${healthResp.uptime_seconds}s` }
    const summaryResp = await getSummary()
    const vehiclesResp = await getVehicles()
    const leadsResp = await getLeads()
    const configResp = await publicConfig()
    const llmResp = await checkLlmConfig()
    await refreshSystemReadiness(false)
    await refreshReleaseGate(false)
    await runPreReleaseCheck(false)
    await runDeliveryPackage(false)
    if (!Array.isArray(vehiclesResp?.vehicles) || !summaryResp?.energy_distribution) throw new Error('接口返回结构异常')
    summary.value = summaryResp
    vehicles.value = vehiclesResp.vehicles
    leads.value = Array.isArray(leadsResp?.leads) ? leadsResp.leads : []
    config.value = { ...(configResp || {}), llm_available: !!llmResp?.available, llm_check: llmResp?.check }
  } catch {
    backendStatus.value = { status: 'offline', detail: staticFallbackMode ? '静态模式：未配置后端地址' : '后端连接失败，已切换兜底数据' }
    vehicles.value = demoVehicles
    summary.value = buildFallbackSummary(demoVehicles)
    leads.value = []
    config.value = { chat_model: '本地兜底模式', api_key_configured: false }
    readiness.value = fallbackReadiness()
    releaseGate.value = fallbackReleaseGate()
    acceptanceReport.value = fallbackAcceptanceReport()
    deliveryPackage.value = fallbackDeliveryPackage()
  }
}

async function runEvaluation() {
  evaluationLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地评估')
    const res = await runRecommendationEvaluation()
    evaluationSummary.value = res.summary || {}
    evaluationCases.value = res.cases || []
    evaluationNote.value = res.obsidian_note || null
    agentRegression.value = await runAgentRegressionEvaluation()
    ElMessage.success(`评估完成：推荐通过率 ${evaluationSummary.value.pass_rate}%，Agent回归 ${agentRegression.value.summary?.pass_rate || 0}%`)
  } catch {
    const res = fallbackEvaluation()
    evaluationSummary.value = res.summary
    evaluationCases.value = res.cases
    evaluationNote.value = res.obsidian_note
    agentRegression.value = fallbackAgentRegression()
    ElMessage.warning('后端评估接口不可用，已展示内置评估样例')
  } finally {
    evaluationLoading.value = false
  }
}

async function refreshFeedback() {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地反馈')
    feedbackSummary.value = await getRecommendationFeedbackSummary()
    feedbackReview.value = await getRecommendationFeedbackReview()
  } catch {
    feedbackSummary.value = fallbackFeedback()
    feedbackReview.value = fallbackFeedbackReview()
  }
}

async function refreshOptimization() {
  optimizationLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地优化建议')
    optimization.value = await getOptimizationInsights()
  } catch {
    optimization.value = fallbackOptimization()
  } finally {
    optimizationLoading.value = false
  }
}

async function refreshRealWorld() {
  realWorldLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用真实数据兜底')
    realWorld.value = await getRealWorldOverview(30)
  } catch {
    realWorld.value = fallbackRealWorld()
  } finally {
    realWorldLoading.value = false
  }
}

async function runRealWorldGovernance() {
  realWorldLoading.value = true
  try {
    const governance = await refreshRealWorldGovernance()
    realWorld.value = { ...realWorld.value, governance }
    ElMessage.success(`真实数据治理完成：评分 ${governance.summary?.quality_score || 0}`)
  } catch {
    realWorld.value = { ...realWorld.value, governance: fallbackRealWorld().governance }
    ElMessage.warning('后端治理接口不可用，已展示本地治理样例')
  } finally {
    realWorldLoading.value = false
  }
}

async function runRealWorldRecommend() {
  realWorldRecLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用真实推荐兜底')
    const res = await recommendRealWorld({ query: query.value, profile: profile.value, top_k: 8 })
    realWorldRecs.value = res.recommendations || []
    ElMessage.success(`真实候选推荐完成：${res.candidate_count || 0} 条候选参与排序`)
  } catch {
    realWorldRecs.value = [
      { brand: 'BMW', model: 'iX xDrive50', vehicle_type: 'SUV', price_min: 535000, price_max: 690000, cltc_range: 630, score: 87.2, data_quality_score: 100, source_type: 'real_world_enriched' },
      { brand: 'Hyundai', model: 'IONIQ 5 AWD', vehicle_type: 'SUV', price_min: 278000, price_max: 365000, cltc_range: 480, score: 84.1, data_quality_score: 100, source_type: 'real_world_enriched' },
      { brand: 'Volkswagen', model: 'ID.7 PRO', vehicle_type: '轿车', price_min: 312000, price_max: 420000, cltc_range: 700, score: 82.8, data_quality_score: 100, source_type: 'real_world_enriched' },
    ]
    ElMessage.warning('后端真实候选推荐接口不可用，已展示本地兜底样例')
  } finally {
    realWorldRecLoading.value = false
  }
}

async function refreshFusedCatalog() {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用融合池兜底')
    fusedCatalog.value = await getFusedCatalog(220)
  } catch {
    fusedCatalog.value = fallbackFused()
  }
}

async function runFusedRecommend() {
  fusedLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用融合推荐兜底')
    const res = await recommendFused({ query: query.value, profile: profile.value, top_k: 8 })
    fusedRecs.value = res.recommendations || []
    fusedCatalog.value = { ...fusedCatalog.value, summary: res.catalog_summary || fusedCatalog.value.summary }
    ElMessage.success(`融合推荐完成：${res.catalog_summary?.total || 0} 条候选参与排序`)
  } catch {
    fusedCatalog.value = fallbackFused()
    fusedRecs.value = [
      { brand: '特斯拉', model: 'Model Y', catalog_source: 'local_curated', vehicle_type: 'SUV', cltc_range: 688, score: 100, reasons: ['本地精选车型，预算和空间匹配'] },
      { brand: '比亚迪', model: '宋PLUS DM-i', catalog_source: 'local_curated', vehicle_type: 'SUV', cltc_range: 1100, score: 99, reasons: ['本地精选车型，补能稳定'] },
      { brand: '宝马', model: 'iX xDrive50', catalog_source: 'real_world_enriched', vehicle_type: 'SUV', cltc_range: 630, score: 92, reasons: ['真实扩展候选，长续航和豪华品牌'] },
    ]
  } finally {
    fusedLoading.value = false
  }
}

function inferScenarioTags() {
  const text = `${query.value} ${(profile.value.concerns || []).join(' ')}`
  const tags = []
  if (/家庭|三口|二胎|孩子/.test(text)) tags.push('家庭SUV')
  if (/无家充|没有家充|充电/.test(text)) tags.push('无家充补能')
  if (/预算|价格|超预算/.test(text)) tags.push('预算敏感')
  if (/智驾|辅助驾驶/.test(text)) tags.push('智驾关注')
  if (/真实|海外|公开数据/.test(text)) tags.push('真实数据查询')
  return tags.length ? tags : ['通用推荐']
}

function inferFeedbackPool(item: any) {
  return explainability.value?.pool_decision?.selected_pool || item.catalog_source || item.source_type || candidatePool.value || 'unknown'
}

async function submitFeedback(item: any, rating: 'positive' | 'negative') {
  const reason = rating === 'positive' ? '推荐符合当前用户画像' : '推荐结果需要继续优化'
  const payload = {
    query: query.value,
    model_name: `${item.brand} ${item.model}`,
    rating,
    reason,
    candidate_pool: inferFeedbackPool(item),
    scenario_tags: inferScenarioTags(),
    profile: profile.value,
    recommendation: item,
  }
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地反馈')
    const res = await createRecommendationFeedback(payload)
    feedbackSummary.value = res.summary
    await refreshFeedback()
    ElMessage.success(`反馈已沉淀到 Obsidian：${res.obsidian_note?.title || payload.model_name}`)
  } catch {
    const now = new Date().toLocaleString('zh-CN', { hour12: false })
    const current = feedbackSummary.value.total ? feedbackSummary.value : fallbackFeedback()
    feedbackSummary.value = { ...current, total: current.total + 1, positive: current.positive + (rating === 'positive' ? 1 : 0), negative: current.negative + (rating === 'negative' ? 1 : 0), positive_rate: Math.round((current.positive + (rating === 'positive' ? 1 : 0)) / (current.total + 1) * 1000) / 10, recent: [{ created_at: now, model_name: payload.model_name, rating, reason }, ...(current.recent || [])] }
    ElMessage.success('后端暂不可用，已在前端记录演示反馈')
  }
}

async function parseProfile() {
  if (!query.value.trim()) return
  profileParsingLoading.value = true
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地解析')
    profilePreview.value = await previewProfile({ query: query.value, profile: profile.value, top_k: 5 })
    profile.value = { ...profile.value, ...profilePreview.value.profile }
  } catch (e: any) {
    profilePreview.value = localProfilePreview()
    profile.value = { ...profile.value, ...profilePreview.value.profile }
    ElMessage.warning(`后端画像接口不可用，已启用本地解析：${e.message || e}`)
  } finally {
    profileParsingLoading.value = false
  }
}

async function submitRecommend() {
  loading.value = true
  agentTrace.value = []
  explainability.value = null
  feedbackPolicy.value = null
  obsidianNote.value = null
  answerHtml.value = '<span class="muted">Agent 正在分析需求，调用工具中...</span>'

  try {
    await parseProfile()
    if (staticFallbackMode) {
      applyLocalRecommend('静态预览模式')
      return
    }
    const res = await recommendAgent({
      query: query.value,
      profile: profile.value,
      top_k: 5,
      use_deep_search: useDeepSearch.value,
      candidate_pool_strategy: candidatePool.value
    })
    recommendations.value = res.recommendations || []
    answerHtml.value = toHtml(res.answer || 'Agent 后端已完成统一推荐编排。')
    agentTrace.value = res.agent_trace || []
    sources.value = res.sources || []
    explainability.value = res.explainability || null
    feedbackPolicy.value = res.feedback_policy || null
    skillTrace.value = res.skill_trace || []
    obsidianNote.value = res.obsidian_note || null
    fusedCatalog.value = { ...fusedCatalog.value, summary: res.catalog_summary || fusedCatalog.value.summary }
    const pool = res.pool_decision?.selected_pool || candidatePool.value
    ElMessage.success(`Agent 推荐完成，后端选择候选池：${pool}`)
    playSectionMotion()
    refresh()
  } catch (e: any) {
    applyLocalRecommend(`请求失败：${e.message || e}`)
  } finally {
    loading.value = false
  }
}

async function askCustomerService() {
  if (!serviceQuestion.value.trim()) return
  const userText = serviceQuestion.value.trim()
  const history = chatMessages.value.slice(-8).map(item => ({ role: item.role, content: item.content }))
  chatMessages.value.push({ role: 'user', content: userText })
  serviceQuestion.value = ''
  serviceLoading.value = true
  try {
    const res = await customerServiceChat(userText, serviceUseWebSearch.value, history)
    serviceAnswerHtml.value = toHtml(res.answer)
    chatMessages.value.push({ role: 'assistant', content: res.answer })
    serviceTrace.value = res.agent_trace
    skillTrace.value = res.skill_trace
    sources.value = res.sources
  } finally {
    serviceLoading.value = false
  }
}

async function submitCompare() {
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地对比')
    const res = await compare({ models: compareModels.value, profile: profile.value })
    compareRows.value = res.result.vehicles
  } catch {
    compareRows.value = localRecommendations().slice(0, 3)
  }
}

function exportCompareCsv() {
  if (!compareRows.value.length) {
    ElMessage.warning('请先生成竞品对比')
    return
  }
  const headers = ['品牌', '车型', '推荐分', '能源', '起售价', '最高价', 'CLTC', '座位数', '预算分', '续航分', '空间分', '补能分', '智驾分', '安全分', '亮点', '短板']
  const rows = compareRows.value.map(x => [
    x.brand, x.model, x.score, x.energy_type, x.price_min, x.price_max, x.cltc_range, x.seats,
    x.budget_score, x.range_score, x.space_score, x.charging_score, x.smart_score, x.safety_score,
    x.highlights, x.weaknesses
  ])
  const csv = [headers, ...rows].map(row => row.map((cell: any) => `"${String(cell ?? '').replaceAll('"', '""')}"`).join(',')).join('\n')
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `竞品对比_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

async function saveLead() {
  const lead = {
    created_at: new Date().toLocaleString('zh-CN', { hour12: false }),
    name: '演示客户',
    budget: profile.value.budget_max,
    city: profile.value.city || '待补充',
    concerns: (profile.value.concerns || []).join('、'),
    intent_level: profile.value.intent_level || '了解中',
    recommended_models: recommendations.value.slice(0, 3).map(x => `${x.brand} ${x.model}`).join('、'),
    next_action: '邀约试驾并确认充电条件'
  }
  try {
    if (staticFallbackMode) throw new Error('静态预览启用本地线索')
    await createLead({ name: lead.name, profile: profile.value, recommended_models: recommendations.value.slice(0, 3).map(x => `${x.brand} ${x.model}`), next_action: lead.next_action })
    await refresh()
    ElMessage.success('线索已保存')
  } catch {
    leads.value = [lead, ...leads.value]
    ElMessage.success('后端暂不可用，已在前端本地保存演示线索')
  }
}

async function rebuildKnowledge() {
  await rebuildRag()
  await refresh()
  ElMessage.success('RAG 索引已重建')
}

async function seedDemo() {
  const res = await seedDemoData()
  summary.value = res.summary
  await refresh()
  ElMessage.success(`已补充 ${res.result.lead_count} 条线索和 ${res.result.recommendation_count} 条推荐日志`)
}

async function clearData() {
  await ElMessageBox.confirm('确认清空线索、推荐日志和会话等运行态数据吗？', '清空运行数据', { type: 'warning' })
  await clearRuntimeData()
  recommendations.value = []
  await refresh()
  ElMessage.success('运行数据已清空')
}

async function runDemo() {
  active.value = 'recommend'
  await submitRecommend()
}

onMounted(async () => {
  playShellMotion()
  await refresh()
  await submitCompare()
  await runEvaluation()
  await refreshFeedback()
  await refreshOptimization()
  await refreshRealWorld()
  await refreshFusedCatalog()
  await runRealWorldRecommend()
  await runFusedRecommend()
})
</script>
