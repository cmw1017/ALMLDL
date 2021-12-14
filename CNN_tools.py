# 가중치 시각화 (#%%를 최상단에 입력하면 모듈을 설치하고 실행됨)
# for w in resnetN.parameters():
#     w = w.data.cpu()
#     print(w.shape)
#     break

# # normalize weights
# min_w = torch.min(w)
# w1 = (-1/(2 * min_w)) * w + 0.5

# # make grid to display it
# grid_size = len(w1)
# x_grid = [w1[i] for i in range(grid_size)]
# x_grid = torchvision.utils.make_grid(x_grid, nrow=8, padding=1)

# plt.imshow(x_grid.permute(1, 2, 0))