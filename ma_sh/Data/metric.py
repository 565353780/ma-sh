Timing_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'Ours'],
    ['Time', '0.5s+9.7s', '8.9s', '51.3s', '77.2s', '39.4s+3.7s'],
]

ShapeNet_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASHMesh', 'MASH', 'GEN+ICP+R10', 'FPS'],
    ['L1-CD\\downarrow', '89.565', '6.381', '17.732', '15.697', '5.450', '4.944\\bf', '6.554', '4.782'],
    ['L2-CD\\downarrow', '429.112', '26.876', '72.523', '66.051', '22.523', '2.268\\bf', '5.945', '1.871'],
    ['FScore\\uparrow', '0.497', '0.988\\bf', '0.812', '0.880', '0.997', '0.998\\bf', '0.951', '0.999'],
    ['D_H\\downarrow', '0.272', '0.023', '0.131', '0.117', '0.019', '0.013\\bf', '0.068', '0.012'],
    ['S_cos\\uparrow', '0.684', '0.974', '0.821', '0.898', '0.980', '0.984\\bf', '0.950', '0.991'],
    ['NIC\\downarrow', '65.023', '19.178', '29.810', '23.035', '18.040', '13.346\\bf', '21.089', '6.230'],
]

ShapeNet_Data = [
    ShapeNet_Data[i][:5] + ShapeNet_Data[i][6:8] for i in range(len(ShapeNet_Data))
]

ShapeNet_NonUniform_8192_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH-Mesh', 'MASH', 'GEN+ICP+R10'],
    ['L1-CD\\downarrow', '93.452', '8.714', '19.031', '17.301', '8.607', '5.932\\bf', '6.723'],
    ['L2-CD\\downarrow', '342.803', '17.809', '53.071', '42.806', '6.687', '6.021\\bf', '6.300'],
    ['FScore\\uparrow', '0.466', '0.892', '0.763', '0.782', '0.917', '0.958\\bf', '0.949'],
    ['D_H\\downarrow', '0.291', '0.048', '0.133', '0.121', '0.032', '0.023\\bf', '0.069'],
    ['S_cos\\uparrow', '0.623', '0.897', '0.820', '0.837', '0.842', '0.972\\bf', '0.963'],
    ['NIC\\downarrow', '67.712', '23.622', '31.796', '29.801', '23.064', '19.823\\bf', '21.107'],
]

ShapeNet_NonUniform_8192_Data = [
    ShapeNet_NonUniform_8192_Data[i][:5] + ShapeNet_NonUniform_8192_Data[i][6:8] for i in range(len(ShapeNet_NonUniform_8192_Data))
]

ShapeNet_NonUniform_4096_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH', 'GEN+ICP+R10'],
    ['L1-CD\\downarrow', '98.017', '12.692', '19.780', '16.568', '9.433\\underline', '6.768\\bf'],
    ['L2-CD\\downarrow', '391.056', '10.667', '38.251', '26.452', '7.739\\underline', '6.919\\bf'],
    ['FScore\\uparrow', '0.436', '0.861', '0.712', '0.774', '0.900\\underline', '0.946\\bf'],
    ['D_H\\downarrow', '0.294', '0.059', '0.149', '0.132', '0.057\\underline', '0.072\\bf'],
    ['S_cos\\uparrow', '0.303', '0.835', '0.807', '0.853\\underline', '0.837', '0.948\\bf'],
    ['NIC\\downarrow', '64.077', '33.378', '33.042', '27.573\\underline', '30.664', '21.120\\bf'],
]

ShapeNet_NonUniform_2048_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH', 'GEN+ICP+R10'],
    ['L1-CD\\downarrow', '89.023', '10.701\\underline', '16.980', '15.523', '12.341', '6.780\\bf'],
    ['L2-CD\\downarrow', '327.094', '19.694', '33.052', '26.824', '16.850\\underline', '7.523\\bf'],
    ['FScore\\uparrow', '0.411', '0.732', '0.759', '0.789\\underline', '0.757', '0.943\\bf'],
    ['D_H\\downarrow',  '0.332', '0.128\\underline', '0.162', '0.154', '0.131', '0.077\\bf'],
    ['S_cos\\uparrow',  '0.312', '0.804\\underline', '0.787', '0.802', '0.798', '0.946\\bf'],
    ['NIC\\downarrow',  '59.071', '24.096\\underline', '31.051', '28.200', '35.323', '21.383\\bf'],
]

ShapeNet_NonUniform_1024_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH', 'MASH-Mesh', 'GEN+ICP+R10'],
    ['L1-CD\\downarrow', '131.059', '14.927', '16.682', '16.593', '18.113', '19.302', '7.172\\bf'],
    ['L2-CD\\downarrow', '462.081', '36.092', '47.041', '33.502', '37.730', '39.503', '8.628\\bf'],
    ['FScore\\uparrow', '0.326', '0.722', '0.698', '0.763', '0.538', '0.516', '0.892\\bf'],
    ['D_H\\downarrow',  '0.442', '0.194', '0.196', '0.192', '0.231', '0.274', '0.116\\bf'],
    ['S_cos\\uparrow',  '0.320', '0.782', '0.798', '0.760', '0.632', '0.617', '0.899\\bf'],
    ['NIC\\downarrow',  '56.711', '31.542', '29.514', '30.298', '40.094', '46.033', '24.046\\bf'],
]

ShapeNet_NonUniform_1024_Data = [
    ShapeNet_NonUniform_1024_Data[i][:6] + ShapeNet_NonUniform_1024_Data[i][7:8] for i in range(len(ShapeNet_NonUniform_1024_Data))
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

KITTI_Data = [
    ['Method', 'SPR+PCA', 'PGR', 'ConvONet', 'ARONet', 'MASH-400', 'MASH\\bf'],
    ['L1-CD\\downarrow', '8.062', '2.211', '3.826', '2.254', '0.180\\underline', '0.101\\bf'],
    ['L2-CD\\downarrow', '64.087', '7.433', '15.627', '4.825', '0.058\\underline', '0.015\\bf'],
    ['FScore(0.1)\\uparrow', '0.017', '0.061', '0.024', '0.037', '0.722\\underline', '0.877\\bf'],
    ['D_H\\downarrow', '33.282', '11.086', '26.028', '13.252', '2.372\\underline', '1.125\\bf'],
]

ShapeNet_Data = [
    ShapeNet_Data[i][:6] + ShapeNet_Data[i][6:8] for i in range(len(ShapeNet_Data))
]
