# 真实数据推荐测试评估报告

生成时间：2026-06-22 14:35:08
真实车型样本：227 条，品牌 30 个，车型 92 个。
测试用例：4 个，通过 4 个，通过率 100.0%。

## 数据概况

- 数据源：open-ev-data/open-ev-data-dataset + OSkrk/Electric-vehicles-EV-Database
- 年份范围：[2010, 2025]
- 车型分布：[('轿车', 174), ('SUV', 43), ('MPV', 7), ('微型车', 3)]

## 用例结果

### 真实样本-长续航纯电通勤：通过

Top 推荐：
- Faraday Future 91｜SUV｜续航 650km｜得分 84.4
- Lucid Air｜轿车｜续航 650km｜得分 82.9
- VW ID Crozz2｜SUV｜续航 500km｜得分 82.3
- Skoda E｜SUV｜续航 500km｜得分 81.9
- Audi Q6｜SUV｜续航 500km｜得分 81.3

检查项：
- ✅ Top1分数达标：实际 84.4，期望 70
- ✅ Top5包含长续航车型：实际 650，期望 450

### 真实样本-家庭SUV：通过

Top 推荐：
- Audi Q6｜SUV｜续航 500km｜得分 95.5
- VW ID Crozz2｜SUV｜续航 500km｜得分 95.5
- Faraday Future 91｜SUV｜续航 650km｜得分 95.4
- Skoda E｜SUV｜续航 500km｜得分 95.4
- Kia Niro｜SUV｜续航 455km｜得分 95.1

检查项：
- ✅ Top1分数达标：实际 95.5，期望 70
- ✅ Top5包含目标车型：实际 ['SUV', 'SUV', 'SUV', 'SUV', 'SUV']，期望 SUV

### 真实样本-多人家庭MPV：通过

Top 推荐：
- Daimler EQV｜MPV｜续航 405km｜得分 100
- VW ID Buzz｜MPV｜续航 500km｜得分 98.2
- Renault Kangoo ZE｜MPV｜续航 220km｜得分 95.3
- Merceds Vito E cell｜MPV｜续航 130km｜得分 90.1
- Citroen Berlingo｜MPV｜续航 110km｜得分 89.0

检查项：
- ✅ Top1分数达标：实际 100，期望 65
- ✅ Top10包含目标车型：实际 ['MPV', 'MPV', 'MPV', 'MPV', 'MPV', 'MPV', 'MPV', 'SUV', 'SUV', 'SUV']，期望 MPV

### 真实样本-豪华社交形象：通过

Top 推荐：
- Hyundai IONIQ 5 Base｜SUV｜续航 507km｜得分 84.0
- Hyundai IONIQ 5 Base｜SUV｜续航 507km｜得分 83.8
- Hyundai IONIQ 5 AWD｜SUV｜续航 480km｜得分 83.7
- Audi Q6｜SUV｜续航 500km｜得分 83.6
- Kia Niro｜SUV｜续航 455km｜得分 83.6

检查项：
- ✅ Top1分数达标：实际 84.0，期望 70
- ✅ Top5包含目标品牌带：实际 ['Hyundai', 'Hyundai', 'Hyundai', 'Audi', 'Kia']，期望 ['Audi', 'BMW', 'Porsche', 'Tesla']