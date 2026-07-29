import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import VChart from 'vue-echarts'
import * as echarts from 'echarts'
import { use } from 'echarts/core'
import { BarChart, FunnelChart, LineChart, PieChart, RadarChart, ScatterChart } from 'echarts/charts'
import { GridComponent, LegendComponent, RadarComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import App from './App.vue'
import './style.css'

use([BarChart, PieChart, LineChart, RadarChart, ScatterChart, FunnelChart, GridComponent, TooltipComponent, LegendComponent, RadarComponent, CanvasRenderer])

const app = createApp(App)
app.use(ElementPlus)
app.component('VChart', VChart)
app.mount('#app')
