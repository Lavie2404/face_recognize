const { createApp, ref, onMounted } = Vue;

createApp({
    setup() {
        const username = ref('');
        const message = ref('');
        const status = ref('');
        const loading = ref(false);
        const scanning = ref(false);
        const action = ref('');
        const toasts = ref([]);
        
        const showToast = (message, type = 'success') => {
            const id = Date.now();
            toasts.value.push({ id, message, type });
            setTimeout(() => {
                toasts.value = toasts.value.filter(t => t.id !== id);
            }, 3000);
        };
        
        const video = ref(null);
        const canvas = ref(null);

        const initCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.value.srcObject = stream;
            } catch (err) {
                console.error("Lỗi truy cập webcam:", err);
                message.value = "Không thể truy cập Webcam. Vui lòng cấp quyền.";
                status.value = "error";
            }
        };

        const captureImage = () => {
            const context = canvas.value.getContext('2d');
            canvas.value.width = video.value.videoWidth;
            canvas.value.height = video.value.videoHeight;
            context.drawImage(video.value, 0, 0, canvas.value.width, canvas.value.height);
            return canvas.value.toDataURL('image/jpeg');
        };

        const handleAction = async (type) => {
            if (type === 'register' && !username.value) {
                showToast("Vui lòng nhập tên người dùng để thiết lập.", "error");
                return;
            }

            action.value = type;
            loading.value = true;
            scanning.value = true;
            message.value = type === 'register' ? "Đang mã hóa khuôn mặt..." : "Đang nhận diện khuôn mặt...";
            status.value = "";

            const imageData = captureImage();
            const apiBase = "http://localhost:8000";
            const endpoint = type === 'register' ? `${apiBase}/register` : `${apiBase}/verify`;

            try {
                const response = await axios.post(endpoint, {
                    username: type === 'register' ? username.value : null,
                    image: imageData
                });

                const data = response.data;
                message.value = data.message;
                status.value = data.status;
                
                if (data.status === 'success') {
                    showToast(data.message, "success");
                    scanning.value = false;
                } else {
                    showToast(data.message, "error");
                }
            } catch (err) {
                console.error("API Error:", err);
                showToast("Lỗi kết nối server.", "error");
            } finally {
                loading.value = false;
                if (status.value !== 'success') {
                    scanning.value = false;
                }
            }
        };

        onMounted(() => {
            initCamera();
        });

        return {
            username,
            message,
            status,
            loading,
            scanning,
            action,
            toasts,
            video,
            canvas,
            handleAction
        };
    }
}).mount('#app');
