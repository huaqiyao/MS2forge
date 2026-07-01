"""
轨迹可视化工具
使用RDKit SVG渲染，确保化学键（特别是芳香键）符合化学规范
"""
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def create_trajectory_html(smiles, trajectory, true_edge_types, halfedge_index, node_types, atomic_numbers,
                           output_path, mol_idx):
    """
    创建交互式HTML轨迹可视化

    使用RDKit SVG绘制，确保化学键（特别是芳香键）符合化学规范

    Args:
        smiles: 真实分子的SMILES字符串（用于显示）
        trajectory: 轨迹列表，每个元素是一个时间步的边类型预测 [num_edges]
        true_edge_types: 真实边类型 [num_edges]
        halfedge_index: 边索引 [2, num_edges]
        node_types: 节点类型索引 [num_nodes]
        atomic_numbers: 原子序数数组 [num_nodes] - 关键！用于构建分子
        output_path: 输出HTML路径
        mol_idx: 分子索引
    """
    try:
        if smiles is None:
            smiles = "N/A"

        # 使用atomic_numbers手动构建分子（确保原子顺序一致）
        mol = Chem.RWMol()
        for atomic_num in atomic_numbers:
            atom = Chem.Atom(int(atomic_num))
            atom.SetNoImplicit(True)  # 禁止自动添加隐式氢
            mol.AddAtom(atom)

        # 添加真实的化学键（用于计算2D坐标）
        for i in range(halfedge_index.shape[1]):
            src, dst = int(halfedge_index[0, i]), int(halfedge_index[1, i])
            bond_type = int(true_edge_types[i])
            if bond_type > 0:  # 有化学键
                try:
                    if bond_type == 1:
                        mol.AddBond(src, dst, Chem.BondType.SINGLE)
                    elif bond_type == 2:
                        mol.AddBond(src, dst, Chem.BondType.DOUBLE)
                    elif bond_type == 3:
                        mol.AddBond(src, dst, Chem.BondType.TRIPLE)
                    elif bond_type == 4:
                        mol.AddBond(src, dst, Chem.BondType.AROMATIC)
                except:
                    pass  # 忽略重复的键

        mol = mol.GetMol()

        # 计算2D坐标
        AllChem.Compute2DCoords(mol)

        # 验证原子数量
        num_atoms = mol.GetNumAtoms()
        if num_atoms != len(atomic_numbers):
            print(f"[错误] 分子 {mol_idx}: 原子数量不匹配！")
            return

        # 生成真实分子的SVG（用于右侧显示）
        true_mol_drawer = Draw.MolDraw2DSVG(600, 600)
        true_mol_drawer.DrawMolecule(mol)
        true_mol_drawer.FinishDrawing()
        true_mol_svg = true_mol_drawer.GetDrawingText()

        # 使用RDKit生成每一步的SVG图像
        trajectory_svgs = []

        for step_idx, pred_edge_types in enumerate(trajectory):
            # 计算准确率
            total_acc = float((pred_edge_types == true_edge_types).mean())

            # 创建分子副本用于绘制
            mol_copy = Chem.RWMol()
            for atomic_num in atomic_numbers:
                atom = Chem.Atom(int(atomic_num))
                atom.SetNoImplicit(True)  # 禁止自动添加隐式氢
                mol_copy.AddAtom(atom)

            # 添加预测的化学键
            for i in range(halfedge_index.shape[1]):
                src, dst = int(halfedge_index[0, i]), int(halfedge_index[1, i])
                bond_type = int(pred_edge_types[i])
                if bond_type > 0:  # 有化学键
                    try:
                        if bond_type == 1:
                            mol_copy.AddBond(src, dst, Chem.BondType.SINGLE)
                        elif bond_type == 2:
                            mol_copy.AddBond(src, dst, Chem.BondType.DOUBLE)
                        elif bond_type == 3:
                            mol_copy.AddBond(src, dst, Chem.BondType.TRIPLE)
                        elif bond_type == 4:
                            mol_copy.AddBond(src, dst, Chem.BondType.AROMATIC)
                    except:
                        pass  # 忽略重复的键

            mol_copy = mol_copy.GetMol()

            # 使用原始分子的2D坐标
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(mol.GetConformer(), assignId=True)

            # 生成SVG
            drawer = Draw.MolDraw2DSVG(600, 600)
            drawer.DrawMolecule(mol_copy)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            trajectory_svgs.append({
                'step': step_idx + 1,
                'accuracy': total_acc,
                'svg': svg
            })

        # 生成HTML（左右布局，学术风格黑白配色）
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分子 {mol_idx} 去噪轨迹</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background-color: #ffffff;
            color: #000000;
            line-height: 1.6;
        }}

        .container {{
            display: flex;
            height: 100vh;
        }}

        .left-panel {{
            flex: 1;
            padding: 30px;
            border-right: 2px solid #000000;
            display: flex;
            flex-direction: column;
            background-color: #fafafa;
        }}

        .right-panel {{
            width: 650px;
            padding: 30px;
            background-color: #ffffff;
            display: flex;
            flex-direction: column;
        }}

        .panel-title {{
            font-size: 16px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #000000;
        }}

        #trajectory-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            margin-bottom: 20px;
        }}

        #true-mol-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            margin-bottom: 15px;
        }}

        .smiles-display {{
            background-color: #f9f9f9;
            border: 1px solid #cccccc;
            padding: 12px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            word-break: break-all;
            text-align: center;
        }}

        .stats {{
            text-align: center;
            padding: 12px;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            margin-bottom: 15px;
            font-family: 'Courier New', monospace;
        }}

        .stats span {{
            display: inline-block;
            margin: 0 15px;
            font-size: 14px;
        }}

        .controls {{
            background-color: #f5f5f5;
            padding: 20px;
            border: 1px solid #cccccc;
        }}

        .slider-container {{
            margin-bottom: 15px;
        }}

        input[type="range"] {{
            width: 100%;
            height: 4px;
            background: #cccccc;
            outline: none;
            -webkit-appearance: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            background: #000000;
            cursor: pointer;
            border-radius: 50%;
        }}

        input[type="range"]::-moz-range-thumb {{
            width: 16px;
            height: 16px;
            background: #000000;
            cursor: pointer;
            border-radius: 50%;
            border: none;
        }}

        .button-group {{
            display: flex;
            gap: 10px;
            justify-content: center;
        }}

        button {{
            padding: 10px 20px;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            background-color: #000000;
            color: #ffffff;
            border: 2px solid #000000;
            transition: all 0.3s;
        }}

        button:hover {{
            background-color: #ffffff;
            color: #000000;
        }}

        button:active {{
            transform: scale(0.98);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <div class="panel-title">Denoising Trajectory</div>

            <div id="trajectory-container"></div>

            <div class="controls">
                <div class="slider-container">
                    <input type="range" id="stepSlider" min="0" max="{len(trajectory)-1}" value="0" step="1">
                </div>
                <div class="stats">
                    <span id="stepInfo">Step: 1/{len(trajectory)}</span>
                    <span>|</span>
                    <span id="accInfo">Accuracy: 0.000</span>
                </div>
                <div class="button-group">
                    <button id="playBtn">Play</button>
                    <button id="prevBtn">Prev</button>
                    <button id="nextBtn">Next</button>
                    <button id="resetBtn">Reset</button>
                </div>
            </div>
        </div>

        <div class="right-panel">
            <div class="panel-title">Ground Truth</div>
            <div id="true-mol-container">{true_mol_svg}</div>
            <div class="smiles-display">{smiles}</div>
        </div>
    </div>

    <script>
        // 轨迹数据（SVG）
        const trajectoryData = {json.dumps(trajectory_svgs)};

        // 当前步骤
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;

        // 显示指定步骤的分子
        function showStep(stepIndex) {{
            const data = trajectoryData[stepIndex];
            const container = document.getElementById('trajectory-container');
            container.innerHTML = data.svg;

            // 更新信息
            document.getElementById('stepInfo').textContent = `Step: ${{data.step}}/${{trajectoryData.length}}`;
            document.getElementById('accInfo').textContent = `Accuracy: ${{data.accuracy.toFixed(3)}}`;

            currentStep = stepIndex;
            document.getElementById('stepSlider').value = stepIndex;
        }}

        // 播放/暂停（播放速度为原来的1.5倍，约133ms）
        document.getElementById('playBtn').addEventListener('click', function() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                this.textContent = 'Play';
                isPlaying = false;
            }} else {{
                this.textContent = 'Pause';
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentStep < trajectoryData.length - 1) {{
                        showStep(currentStep + 1);
                    }} else {{
                        clearInterval(playInterval);
                        document.getElementById('playBtn').textContent = 'Play';
                        isPlaying = false;
                    }}
                }}, 133);
            }}
        }});

        // 上一步
        document.getElementById('prevBtn').addEventListener('click', function() {{
            if (currentStep > 0) {{
                showStep(currentStep - 1);
            }}
        }});

        // 下一步
        document.getElementById('nextBtn').addEventListener('click', function() {{
            if (currentStep < trajectoryData.length - 1) {{
                showStep(currentStep + 1);
            }}
        }});

        // 重置
        document.getElementById('resetBtn').addEventListener('click', function() {{
            showStep(0);
        }});

        // 滑块
        document.getElementById('stepSlider').addEventListener('input', function() {{
            showStep(parseInt(this.value));
        }});

        // 初始化
        showStep(0);
    </script>
</body>
</html>
"""

        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[成功] 轨迹HTML已保存: {output_path}")

    except Exception as e:
        print(f"[错误] 生成HTML失败: {e}")
        import traceback
        traceback.print_exc()
