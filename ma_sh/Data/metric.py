Timing_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'Ours'],
    ['Time', '0.5s+9.7s', '8.9s', '51.3s', '77.2s', '39.4s+3.7s'],
]

ShapeNet_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASHMesh', 'MASH', 'FPS'],
    ['L1-CD\\downarrow', '89.565', '6.381', '17.732', '15.697', '5.450', '4.944', '4.782'],
    ['L2-CD\\downarrow', '429.112', '26.876', '72.523', '66.051', '22.523', '2.268', '1.871'],
    ['FScore\\uparrow', '0.497', '0.988', '0.812', '0.880', '0.997', '0.998', '0.999'],
    ['D_H\\downarrow', '0.272', '0.023', '0.131', '0.117', '0.019', '0.013', '0.012'],
    ['S_cos\\uparrow', '0.684', '0.974', '0.821', '0.898', '0.980', '0.984', '0.991'],
    ['NIC\\downarrow', '65.023', '19.178', '29.810', '23.035', '18.040', '13.346', '6.230'],
]

ShapeNet_NonUniform_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASHMesh', 'MASH'],
    ['L1-CD\\downarrow', '98.017', '12.692\\underline', '19.780', '16.568', '15.440', '9.433\\bf'],
    ['L2-CD\\downarrow', '391.056', '10.667\\underline', '38.251', '26.452', '44.769', '7.739\\bf'],
    ['FScore\\uparrow', '0.436', '0.861\\underline', '0.712', '0.774', '0.828', '0.900\\bf'],
    ['D_H\\downarrow', '0.294', '0.059\\underline', '0.149', '0.132', '0.065', '0.057\\bf'],
    ['S_cos\\uparrow', '0.303', '0.835', '0.807', '0.853', '0.849', '0.837\\underline'],
    ['NIC\\downarrow', '64.077', '33.378', '33.042', '27.573', '29.431', '30.664\\underline'],
]

# new
ShapeNet_NonUniform_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH'],
    ['L1-CD\\downarrow', '98.017', '12.692\\underline', '19.780', '16.568', '9.433\\bf'],
    ['L2-CD\\downarrow', '391.056', '10.667\\underline', '38.251', '26.452', '7.739\\bf'],
    ['FScore\\uparrow', '0.436', '0.861\\underline', '0.712', '0.774', '0.900\\bf'],
    ['D_H\\downarrow', '0.294', '0.059\\underline', '0.149', '0.132', '0.057\\bf'],
    ['S_cos\\uparrow', '0.303', '0.835', '0.807', '0.853\\bf', '0.837\\underline'],
    ['NIC\\downarrow', '64.077', '33.378', '33.042', '27.573\\bf', '30.664\\underline'],
]

ShapeNet_NonUniform_2048_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH', 'GEN+ABBAlign-MEAN', 'GEN+ABBAlign-BEST'],
    ['L1-CD\\downarrow', '89.023', '10.701\\underline', '16.980', '15.523', '12.341', '17.081', '6.893\\bf'],
    ['L2-CD\\downarrow', '327.094', '19.694', '33.052', '26.824', '16.850\\underline', '27.324', '11.271\\bf'],
    ['FScore\\uparrow', '0.411', '0.732', '0.759', '0.789\\underline', '0.757', '0.628', '0.913\\bf'],
    ['D_H\\downarrow',  '0.332', '0.128\\underline', '0.162', '0.154', '0.131', '0.188', '0.089\\bf'],
    ['S_cos\\uparrow',  '0.312', '0.804', '0.787', '0.802', '0.798', '0.842\\underline', '0.945\\bf'],
    ['NIC\\downarrow',  '59.071', '24.096', '31.051', '28.200', '35.323', '23.701\\underline', '21.607\\bf'],
]

ShapeNet_NonUniform_1024_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH', 'GEN+ICP'],
    ['L1-CD\\downarrow', '', '14.927', '16.682', '16.593', '18.113', ''],
    ['L2-CD\\downarrow', '', '36.092', '47.041', '33.502', '37.730', ''],
    ['FScore\\uparrow', '', '0.722', '0.698', '0.763', '0.538', ''],
    ['D_H\\downarrow',  '', '', '0.196', '0.192', '', ''],
    ['S_cos\\uparrow',  '', '', '', '', '', ''],
    ['NIC\\downarrow',  '', '28.542', '', '30.298', '40.094', ''],
]

Thingi10K_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASHMesh', 'MASH', 'MASH-1600anc'],
    ['L1-CD\\downarrow', '54.677', '4.862', '18.692', '17.472', '6.202', '5.618', '3.353'],
    ['L2-CD\\downarrow', '218.907', '1.443', '29.132', '27.770', '4.718', '2.599', '0.843'],
    ['FScore\\uparrow', '0.552', '0.999', '0.680', '0.682', '0.926', '0.987', '0.999'],
    ['D_H\\downarrow', '0.293', '0.019', '0.176', '0.171', '0.027', '0.024', '0.013'],
    ['S_cos\\uparrow', '0.403', '0.882', '0.789', '0.783', '0.790', '0.823', '0.891'],
    ['NIC\\downarrow', '68.107', '34.144', '46.807', '41.533', '39.089', '35.672', '27.146'],
]

Thingi10K_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH-400', 'MASH-1600'],
    ['L1-CD\\downarrow', '54.677', '4.862\\underline', '18.692', '17.472', '5.618', '3.353\\bf'],
    ['L2-CD\\downarrow', '218.907', '1.443\\underline', '29.132', '27.770', '2.599', '0.843\\bf'],
    ['FScore\\uparrow', '0.552', '0.999\\bf', '0.680', '0.682', '0.987', '0.999\\bf'],
    ['D_H\\downarrow', '0.293', '0.019\\underline', '0.176', '0.171', '0.024', '0.013\\bf'],
    ['S_cos\\uparrow', '0.403', '0.882\\underline', '0.789', '0.783', '0.823', '0.891\\bf'],
    ['NIC\\downarrow', '68.107', '34.144\\underline', '46.807', '41.533', '35.672', '27.146\\bf'],
]

SAMPLE_Data = [
    ['Method', 'L1-CD\\downarrow', 'L2-CD\\downarrow', 'FScore\\uparrow', 'D_H\\downarrow', 'S_cos\\uparrow', 'NIC\\downarrow'],
    ['SIMPLE_MASH', '4.980', '2.463', '0.999', '0.018', '0.985', '12.507'],
    ['MASH', '4.656', '2.172', '0.999', '0.013', '0.987', '11.183'],
]

Coverage_Data = [
    ['', 'L=2' , 'L=3', 'L=4', 'L=5', 'L=6'],
    ['L_c', '6.569' , '5.926', '5.714', '5.444', '5.268'],
]
