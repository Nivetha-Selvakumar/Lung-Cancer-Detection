export function generateSampleCT(type: 'normal' | 'benign' | 'malignant'): Promise<File> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      throw new Error('Canvas context unavailable');
    }

    // Background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, 224, 224);

    // Body Contour
    ctx.fillStyle = '#2a2a2a';
    ctx.beginPath();
    ctx.ellipse(112, 112, 95, 80, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Left Lung Parenchyma
    ctx.fillStyle = '#050505';
    ctx.beginPath();
    ctx.ellipse(75, 112, 32, 50, -0.1, 0, 2 * Math.PI);
    ctx.fill();

    // Right Lung Parenchyma
    ctx.beginPath();
    ctx.ellipse(149, 112, 32, 50, 0.1, 0, 2 * Math.PI);
    ctx.fill();

    if (type === 'malignant') {
      // Irregular high-density opacity (Spicular Nodule)
      ctx.fillStyle = '#d8d8d8';
      ctx.beginPath();
      ctx.arc(75, 125, 14, 0, 2 * Math.PI);
      ctx.fill();
    } else if (type === 'benign') {
      // Smooth calcified nodule
      ctx.fillStyle = '#a0a0a0';
      ctx.beginPath();
      ctx.arc(150, 100, 7, 0, 2 * Math.PI);
      ctx.fill();
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `sample_${type}.png`, { type: 'image/png' });
        resolve(file);
      }
    }, 'image/png');
  });
}
