 # Script 6: Noise2Void, Noise2Self, and Neighbor2Neighbor training
# --- MODULE 2: NOISE2VOID (N2V) SEISMIC ---
print("\n[MODULE 2] Training Noise2Void (N2V) model on geological faults...")
log_n2v_ptk3 = "experimento_N2V_ptk3_logs.csv"
if os.path.exists(log_n2v_ptk3): os.remove(log_n2v_ptk3)

class Red_N2V_PtK3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ciega = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.red_interna = nn.Sequential(
            nn.LeakyReLU(0.1), nn.Conv2d(16, 16, kernel_size=1),
            nn.LeakyReLU(0.1), nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x):
        with torch.no_grad(): self.conv_ciega.weight[:, :, 1, 1] = 0.0
        return self.red_interna(self.conv_ciega(x))

model_n2v = Red_N2V_PtK3().to(device)
opt_n2v = optim.Adam(model_n2v.parameters(), lr=0.002)
criterion = nn.MSELoss()

for epoch in range(120):
    model_n2v.train()
    loss_epoch = 0.0
    for n1, _, _ in dataloader_ptk3:
        n1 = n1.to(device)
        out = model_n2v(n1)
        loss = criterion(out, n1)
        
        opt_n2v.zero_grad()
        loss.backward()
        opt_n2v.step()
        loss_epoch += loss.item() * n1.size(0)
        
    model_n2v.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev in eval_loader_ptk3:
            d_np = model_n2v(n1_ev.to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk3(log_n2v_ptk3, epoch+1, loss_epoch/len(dataset_ptk3), 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2V PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported: '{log_n2v_ptk3}'")


# --- MODULE 3: NOISE2SELF (N2S) SEISMIC ---
print("\n[MODULE 3] Training Noise2Self (N2S) model on geological faults...")
log_n2s_ptk3 = "experimento_N2S_ptk3_logs.csv"
if os.path.exists(log_n2s_ptk3): os.remove(log_n2s_ptk3)

model_n2s = Red_N2V_PtK3().to(device)
opt_n2s = optim.Adam(model_n2s.parameters(), lr=0.002, weight_decay=1e-6)

for epoch in range(120):
    model_n2s.train()
    loss_epoch = 0.0
    for n1, _, _ in dataloader_ptk3:
        n1 = n1.to(device)
        out = model_n2s(n1)
        loss = criterion(out, n1)
        
        opt_n2s.zero_grad()
        loss.backward()
        opt_n2s.step()
        loss_epoch += loss.item() * n1.size(0)
        
    model_n2s.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev in eval_loader_ptk3:
            d_np = model_n2s(n1_ev.to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk3(log_n2s_ptk3, epoch+1, loss_epoch/len(dataset_ptk3), 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2S PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported: '{log_n2s_ptk3}'")


# --- MODULE 4: NEIGHBOR2NEIGHBOR (N2N) SEISMIC ---
print("\n[MODULE 4] Training Neighbor2Neighbor (N2N) model on geological faults...")
log_n2n_ptk3 = "experimento_N2N_ptk3_logs.csv"
if os.path.exists(log_n2n_ptk3): os.remove(log_n2n_ptk3)

class RedEstandarN2N_PtK3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model_n2n = RedEstandarN2N_PtK3().to(device)
opt_n2n = optim.Adam(model_n2n.parameters(), lr=0.002)

for epoch in range(120):
    model_n2n.train()
    loss_epoch = 0.0
    for n1, _, _ in dataloader_ptk3:
        sub_img1 = n1[:, :, 0::2, 0::2]
        sub_img2 = n1[:, :, 0::2, 1::2]
        
        sub_img1_up = F.interpolate(sub_img1, size=(64, 64), mode='bilinear', align_corners=False)
        sub_img2_up = F.interpolate(sub_img2, size=(64, 64), mode='bilinear', align_corners=False)
        
        out_sub1 = model_n2n(sub_img1_up.to(device))
        loss = criterion(out_sub1, sub_img2_up.to(device))
        
        opt_n2n.zero_grad()
        loss.backward()
        opt_n2n.step()
        loss_epoch += loss.item() * n1.size(0)
        
    model_n2n.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev in eval_loader_ptk3:
            d_np = model_n2n(n1_ev.to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk3(log_n2n_ptk3, epoch+1, loss_epoch/len(dataset_ptk3), 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2N PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported: '{log_n2n_ptk3}'")
print("\nSOTA BENCHMARK PROCESS COMPLETED FOR THE SEISMIC PTK3 SCENARIO!")
