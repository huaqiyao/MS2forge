



import json
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def create_trajectory_html(smiles, trajectory, true_edge_types, halfedge_index, node_types, atomic_numbers,
                           output_path, mol_idx):















    try:
        if smiles is None:
            smiles = "N/A"


        mol = Chem.RWMol()
        for atomic_num in atomic_numbers:
            atom = Chem.Atom(int(atomic_num))
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)


        for i in range(halfedge_index.shape[1]):
            src, dst = int(halfedge_index[0, i]), int(halfedge_index[1, i])
            bond_type = int(true_edge_types[i])
            if bond_type > 0:
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
                    pass

        mol = mol.GetMol()


        AllChem.Compute2DCoords(mol)


        num_atoms = mol.GetNumAtoms()
        if num_atoms != len(atomic_numbers):
            print(f"[ERROR] molecule {mol_idx}: Atom-count mismatch！")
            return


        true_mol_drawer = Draw.MolDraw2DSVG(600, 600)
        true_mol_drawer.DrawMolecule(mol)
        true_mol_drawer.FinishDrawing()
        true_mol_svg = true_mol_drawer.GetDrawingText()


        trajectory_svgs = []

        for step_idx, pred_edge_types in enumerate(trajectory):

            total_acc = float((pred_edge_types == true_edge_types).mean())


            mol_copy = Chem.RWMol()
            for atomic_num in atomic_numbers:
                atom = Chem.Atom(int(atomic_num))
                atom.SetNoImplicit(True)
                mol_copy.AddAtom(atom)


            for i in range(halfedge_index.shape[1]):
                src, dst = int(halfedge_index[0, i]), int(halfedge_index[1, i])
                bond_type = int(pred_edge_types[i])
                if bond_type > 0:
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
                        pass

            mol_copy = mol_copy.GetMol()


            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(mol.GetConformer(), assignId=True)


            drawer = Draw.MolDraw2DSVG(600, 600)
            drawer.DrawMolecule(mol_copy)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            trajectory_svgs.append({
                'step': step_idx + 1,
                'accuracy': total_acc,
                'svg': svg
            })


        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>molecule {mol_idx} denoising trajectory</title>
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
        // trajectory data(SVG)
        const trajectoryData = {json.dumps(trajectory_svgs)};

        // current step
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;

        // display the molecule at the selected step
        function showStep(stepIndex) {{
            const data = trajectoryData[stepIndex];
            const container = document.getElementById('trajectory-container');
            container.innerHTML = data.svg;

            // update information
            document.getElementById('stepInfo').textContent = `Step: ${{data.step}}/${{trajectoryData.length}}`;
            document.getElementById('accInfo').textContent = `Accuracy: ${{data.accuracy.toFixed(3)}}`;

            currentStep = stepIndex;
            document.getElementById('stepSlider').value = stepIndex;
        }}

        // play/pause(playback speed is 1.5x, approximately 133ms)
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

        // previous step
        document.getElementById('prevBtn').addEventListener('click', function() {{
            if (currentStep > 0) {{
                showStep(currentStep - 1);
            }}
        }});

        // next step
        document.getElementById('nextBtn').addEventListener('click', function() {{
            if (currentStep < trajectoryData.length - 1) {{
                showStep(currentStep + 1);
            }}
        }});

        // reset
        document.getElementById('resetBtn').addEventListener('click', function() {{
            showStep(0);
        }});

        // slider
        document.getElementById('stepSlider').addEventListener('input', function() {{
            showStep(parseInt(this.value));
        }});

        // initialization
        showStep(0);
    </script>
</body>
</html>
"""


        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[successful] trajectoryHTMLsaved: {output_path}")

    except Exception as e:
        print(f"[ERROR] generateHTMLfailed: {e}")
        import traceback
        traceback.print_exc()
