## 3. Phương pháp: AR-GSE

### 3.1 Thiết lập & ký hiệu

Cho $(X, Y) \sim \mathcal{D}$ với $Y = \{1, \dots, C\}$. Các lớp được phân nhóm $\{G_k\}_{k=1}^K$; ký hiệu $[y] = k$ nếu $y \in G_k$.

Có $E$ chuyên gia (experts) $f^{(e)}$ cho hậu nghiệm đã hiệu chỉnh:

$$
p^{(e)}(y \mid x) \in \Delta_{C-1} \quad (e = 1..E)
$$

Một hàm gating $w_\phi : X \to \Delta_{E-1}$ sinh trọng số trộn $w_\phi(x)_e \geq 0$ và $\sum_e w_\phi(x)_e = 1$.

Hậu nghiệm trộn:

$$
\tilde{\eta}_y(x) = \sum_{e=1}^E w_\phi(x)_e \cdot p^{(e)}(y \mid x)
$$

Bộ phân loại chọn lọc là cặp $(h, r)$ với:

- $h: X \to Y$
- $r: X \to \{0, 1\}$ (nhận nếu $r=0$, từ chối nếu $r=1$)

Lỗi chọn lọc theo nhóm:

$$
e_k(h, r) = \Pr(Y \neq h(X) \mid r(X) = 0, Y \in G_k)
$$

Hai mục tiêu:

- **Balanced**:

$$
R_{\text{bal}}(h, r) = \frac{1}{K} \sum_{k=1}^K e_k(h, r) + c \Pr(r = 1)
$$

- **Worst-group**:

$$
R_{\max}(h, r) = \max_k e_k(h, r) + c \Pr(r = 1)
$$

Tham số $c \geq 0$ là chi phí từ chối.

---

### 3.2 Họ quy tắc tuyến tính–ngưỡng (linear-threshold)

Tham số hoá bởi $(\alpha, \mu) \in (0, \infty)^K \times \mathbb{R}^K$:

- **Phân loại**:

$$
h_\alpha(x) = \arg\max_y \tilde{\eta}_y(x) \cdot \alpha_{[y]}
$$

- **Từ chối**:

$$
r_{\alpha, \mu}(x) = \mathbf{1} \left\{
\max_y \tilde{\eta}_y(x) \cdot \alpha_{[y]} < \sum_{y'} \left( \frac{1}{\alpha_{[y']}} - \mu_{[y']} \right) \tilde{\eta}_{y'}(x) - c
\right\}
$$

**Trực giác**:

- $\alpha$: tái cân bằng điểm theo nhóm.
- $\mu$: dịch ngưỡng chọn lọc.
- Với $K = 2$, có thể dùng $\lambda = \mu_1 - \mu_2$.

---

### 3.3 Mục tiêu nhạy chi phí & ràng buộc chấp nhận theo nhóm

Tổng quát hóa mục tiêu với $\beta \in \Delta_{K-1}$:

$$
R_\beta(h, r) = \sum_{k=1}^K \beta_k e_k(h, r) + c \Pr(r = 1)
$$

Ràng buộc chấp nhận theo nhóm:

$$
K \cdot \Pr(r(X) = 0, Y \in G_k) = \alpha_k, \quad \forall k = 1..K
$$

---

### 3.4 Huấn luyện end-to-end ăn khớp rủi ro (risk-aligned)

Học $(\phi, \alpha, \mu)$ trực tiếp với biến chấp nhận "mềm":

- **Biên chọn lọc**:

$$
m_{\alpha, \mu}(x) = \max_y \tilde{\eta}_y(x) \alpha_{[y]} - \left( \sum_{y'} \left( \frac{1}{\alpha_{[y']}} - \mu_{[y']} \right) \tilde{\eta}_{y'}(x) - c \right)
$$

- **Soft indicator**:

$$
s_\tau(x) = \sigma(\tau \cdot m_{\alpha, \mu}(x))
$$

Bài toán tối ưu **primal–dual**:

$$
\min_{\phi, \alpha > 0, \mu} \sum_{k=1}^K \beta_k \alpha_k \mathbb{E}[\ell_{\text{cls}}(x, y; \tilde{\eta}) \cdot s_\tau(x) \cdot \mathbf{1}_{y \in G_k}] + c \mathbb{E}[1 - s_\tau(x)] + \sum_{k=1}^K \lambda_k \left( \alpha_k - K \mathbb{E}[s_\tau(x) \cdot \mathbf{1}_{y \in G_k}] \right)
$$

- $\ell_{\text{cls}}$: surrogate như CE hoặc $1 - \tilde{\eta}_y$
- $\lambda_k \geq 0$: biến dual
- Cập nhật dual:

$$
\lambda_k \leftarrow \left[ \lambda_k + \rho \left( \alpha_k - K \hat{\mathbb{E}}^B [s_\tau \cdot \mathbf{1}_{y \in G_k}] \right) \right]_+
$$

Warm-up $\tau$: từ 2 → 10 trong 30 epoch.

---

### 3.5 Định lý & bảo đảm

- **Định lý 1**: Họ tuyến tính-ngưỡng đủ để đạt tối ưu dưới ràng buộc chấp nhận.
- **Mệnh đề 1**: Nếu $\tilde{\eta} = \eta$ và $\tau \to \infty$, thì minimizer khôi phục quyết định cứng.
- **Mệnh đề 2**: Với step-size Robbins–Monro, cập nhật primal–dual hội tụ, vi phạm ràng buộc giảm $O(1/T)$.

---

### 3.6 Thiết kế gating & mở rộng

Đặc trưng vào gating: entropy của expert, mass top-k, phân tán top-k, pairwise KL/cosine, và random projection.

- **Sparsity / Hard-routing**: phạt entropy hoặc $\ell_1$ để $w_\phi(x)$ gần one-hot.

---

### 3.7 Worst-group: thích nghi trọng số nhóm

Cập nhật $\beta$ bằng Exponentiated Gradient (EG):

$$
\beta_k \propto \beta_k \cdot \exp(\xi \cdot \hat{e}_k), \quad \sum_k \beta_k = 1
$$

---

### 3.8 Thuật toán

#### Thuật toán 1 — AR-GSE (Balanced / Cost-Sensitive)

- Input: experts $\{p^{(e)}\}$, tuning set $V$, $\beta$ (balanced: $\frac{1}{K}$).
- Init: $\phi$ (MLP nhỏ), $\alpha_k \leftarrow 1$, $\mu \leftarrow 0$, $\lambda \leftarrow 0$, $\tau \leftarrow 2$.

**Loop qua epoch**:

1. Tính $w_\phi(x) = \text{softmax}(\text{MLP}(x))$
2. Tính $\tilde{\eta} = \sum_e w_\phi \cdot p^{(e)}$
3. Tính $m_{\alpha, \mu}(x)$ và $s_\tau(x) = \sigma(\tau m)$
4. Loss:

$$
L = \sum_k \beta_k \alpha_k \mathbb{E}[\ell_{\text{cls}} \cdot s_\tau \cdot \mathbf{1}_{y \in G_k}] + c \mathbb{E}[1 - s_\tau] + \sum_k \lambda_k \left( \alpha_k - K \mathbb{E}[s_\tau \cdot \mathbf{1}_{y \in G_k}] \right) + \lambda_{\text{ent}} \mathbb{E}[H(w_\phi)]
$$

5. **Primal step**: SGD với $\phi$, $\alpha$, $\mu$
6. **Dual step**: cập nhật $\lambda_k$

---

#### Thuật toán 2 — AR-GSE-Worst (EG ngoài)

- Chạy thuật toán 1
- Sau mỗi epoch, cập nhật $\beta$ bằng EG với $\hat{e}_k$

---

#### (Tuỳ chọn) Thuật toán 3 — Conformal calibration theo nhóm

- Dùng tập calibrate để lấy ngưỡng phân vị $t_k$ cho $m_{\alpha, \mu}(x)$
- Nhận nếu $m_{\alpha, \mu}(x) \geq \hat{t}_k(x)$

---

### 3.9 Thực hành & độ phức tạp

- Hyper mặc định:
  - $\eta_\phi = 1e{-3}$
  - $\eta_\alpha = \eta_\mu = 5e{-3}$
  - $\rho = 1e{-2}$, $\tau: 2 \to 10$ (30 epoch)
  - $\lambda_{\text{ent}} = 1e{-3}$, $\epsilon = 1e{-3}$

- Suy luận: $\mathcal{O}(E \cdot C)$; với hard-routing ~ chi phí 1 expert.

- Khả năng mở rộng: iNat (nghìn lớp) dùng gating không phụ thuộc $C$.

---

### 3.10 Gợi ý triển khai

- **Experts** (Stage A): train + temperature scaling
- **AR-GSE** (Stage duy nhất): `train_argse.py` chạy Thuật toán 1/2
- Không cần quét $\mu$, không cần fixed-point hay vòng lặp $T$
- Đầu ra: $(w_\phi, \alpha, \mu)$ + log $\lambda$, coverage/group error, RC-curve, AURC.
- (Tuỳ chọn) conformal để có bảo đảm xác suất hữu hạn.
