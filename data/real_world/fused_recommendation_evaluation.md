# 融合候选池推荐评估报告

生成时间：2026-06-22 14:41:36
融合候选池：{'total': 229, 'local_count': 38, 'real_count': 191, 'dedup_skipped': 29, 'brand_count': 46, 'brand_distribution': [('宝马', 42), ('奥迪', 32), ('大众', 31), ('现代', 26), ('标致', 16), ('雷诺', 7), ('特斯拉', 6), ('日产', 5), ('问界', 4), ('奔驰', 4), ('Daimler', 4), ('福特', 4), ('极氪', 3), ('蔚来', 3), ('雪铁龙', 3), ('比亚迪', 2)], 'vehicle_type_distribution': [('轿车', 158), ('SUV', 59), ('MPV', 7), ('微型车', 3), ('猎装车', 1), ('旅行车', 1)]}
通过率：100.0%

## 融合池-中文家庭SUV：通过

- 特斯拉 Model Y｜local_curated｜续航 688km｜100分
- 比亚迪 宋PLUS DM-i｜local_curated｜续航 1100km｜100分
- 理想 L6｜local_curated｜续航 1390km｜100分
- 问界 M7｜local_curated｜续航 1300km｜100分
- 埃安 AION Y｜local_curated｜续航 610km｜100分
- ✅ 融合候选池规模：实际 229，期望 200
- ✅ Top8包含本地精选：实际 ['local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'real_world_enriched']，期望 local_curated
- ✅ Top8包含真实扩展：实际 ['local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'local_curated', 'real_world_enriched']，期望 real_world_enriched

## 融合池-豪华社交：通过

- 特斯拉 Model 3｜local_curated｜续航 830km｜100分
- 理想 L7｜local_curated｜续航 1315km｜100分
- 问界 M9｜local_curated｜续航 1402km｜100分
- 奔驰 E300L｜local_curated｜续航 0km｜100分
- 宝马 530Li｜local_curated｜续航 0km｜100分
- ✅ Top8包含目标品牌：实际 ['特斯拉', '理想', '问界', '奔驰', '宝马', '奥迪', '蔚来', '奥迪']，期望 ['宝马', '奔驰', '奥迪', '保时捷', '特斯拉']

## 融合池-有家充长续航：通过

- 小鹏 G6｜local_curated｜续航 755km｜94.4分
- 腾势 N7｜local_curated｜续航 702km｜94.2分
- 智界 R7｜local_curated｜续航 802km｜93.8分
- 极氪 007｜local_curated｜续航 870km｜92.9分
- 智界 S7｜local_curated｜续航 855km｜92.9分
- ✅ Top8包含长续航：实际 1390，期望 550
