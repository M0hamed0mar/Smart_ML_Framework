// Enhanced Theme Management with Smooth Transitions
document.addEventListener('DOMContentLoaded', function () {
    // Initialize theme with smooth transition
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTimeout(() => {
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeToggleIcon(savedTheme);
        updateMetaThemeColor(savedTheme);
    }, 100);

    // Enhanced theme toggle functionality
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
                toggleTheme();
            }, 150);
        });

        // Add hover effects
        themeToggle.addEventListener('mouseenter', function () {
            this.style.transform = 'scale(1.1)';
        });

        themeToggle.addEventListener('mouseleave', function () {
            this.style.transform = '';
        });
    }

    // Enhanced file input with drag & drop
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        const parentLabel = input.closest('label');
        if (parentLabel && parentLabel.classList.contains('file-upload-label')) {
            // Drag and drop functionality
            parentLabel.addEventListener('dragover', function (e) {
                e.preventDefault();
                this.style.borderColor = 'var(--accent)';
                this.style.background = 'rgba(var(--accent-rgb), 0.1)';
                this.style.transform = 'scale(1.02)';
            });

            parentLabel.addEventListener('dragleave', function () {
                this.style.borderColor = 'var(--border)';
                this.style.background = 'var(--card)';
                this.style.transform = '';
            });

            parentLabel.addEventListener('drop', function (e) {
                e.preventDefault();
                this.style.borderColor = 'var(--border)';
                this.style.background = 'var(--card)';
                this.style.transform = '';

                if (e.dataTransfer.files.length > 0) {
                    input.files = e.dataTransfer.files;
                    // Trigger change event
                    const event = new Event('change', { bubbles: true });
                    input.dispatchEvent(event);
                    showNotification('File ready for upload!', 'success');
                }
            });
        }
    });

    // Enhanced form validation with animations
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const requiredFields = form.querySelectorAll('[required]');
            let valid = true;

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    valid = false;
                    highlightField(field, false);
                    // Shake animation for invalid fields
                    field.style.animation = 'shake 0.5s ease';
                    setTimeout(() => field.style.animation = '', 500);
                } else {
                    highlightField(field, true);
                }
            });

            if (!valid) {
                e.preventDefault();
                showNotification('Please fill all required fields', 'error');
            }
        });
    });

    // Enhanced auto-dismiss flash messages
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        const closeBtn = message.querySelector('.flash-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                dismissFlashMessage(message);
            });
        }

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            dismissFlashMessage(message);
        }, 5000);
    });

    // Initialize tooltips with animations
    initTooltips();

    // Initialize tabs with smooth transitions
    initTabs();

    // Enhanced smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // Initialize interactive elements
    initInteractiveElements();

    // Add scroll progress indicator
    initScrollProgress();

    // Initialize analytics (optional)
    initAnalytics();

    console.log('AutoML Studio initialized successfully!');
});

// Enhanced Theme functions
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    // Smooth theme transition
    document.documentElement.style.transition = 'all 0.5s ease';
    document.documentElement.setAttribute('data-theme', newTheme);

    setTimeout(() => {
        document.documentElement.style.transition = '';
    }, 500);

    localStorage.setItem('theme', newTheme);
    updateThemeToggleIcon(newTheme);
    updateMetaThemeColor(newTheme);

    showNotification(`Switched to ${newTheme} mode`, 'success');
}

function updateThemeToggleIcon(theme) {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        const icon = themeToggle.querySelector('i');
        if (icon) {
            icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
        }
        themeToggle.setAttribute('aria-label', `Switch to ${theme === 'light' ? 'dark' : 'light'} mode`);
    }
}

function updateMetaThemeColor(theme) {
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
        metaThemeColor.setAttribute('content', theme === 'light' ? '#4e73df' : '#1e1e1e');
    }
}

// Enhanced Helper functions
function highlightField(field, isValid) {
    field.style.borderColor = isValid ? 'var(--success)' : 'var(--danger)';
    field.style.boxShadow = isValid ?
        '0 0 0 2px rgba(var(--success-rgb), 0.1)' :
        '0 0 0 2px rgba(var(--danger-rgb), 0.1)';
}

function showNotification(message, type = 'info', duration = 5000) {
    const flashContainer = document.querySelector('.flash-messages') || createFlashContainer();
    const notification = document.createElement('div');
    notification.className = `flash-message ${type}`;

    const iconMap = {
        'error': 'exclamation-circle',
        'success': 'check-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };

    notification.innerHTML = `
        <i class="fas fa-${iconMap[type] || 'info-circle'}"></i>
        <span>${message}</span>
        <button class="flash-close">&times;</button>
    `;

    notification.style.animation = 'slideIn 0.3s ease';
    flashContainer.appendChild(notification);

    // Add close event
    const closeBtn = notification.querySelector('.flash-close');
    closeBtn.addEventListener('click', () => {
        dismissFlashMessage(notification);
    });

    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => {
            dismissFlashMessage(notification);
        }, duration);
    }

    return notification;
}

function dismissFlashMessage(message) {
    message.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
        if (message.parentNode) {
            message.parentNode.removeChild(message);
        }
    }, 300);
}

function createFlashContainer() {
    const container = document.createElement('div');
    container.className = 'flash-messages';
    document.body.appendChild(container);
    return container;
}

// Enhanced tooltips with animations
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseenter', function (e) {
            const tooltipText = this.getAttribute('data-tooltip');
            const tooltipElem = document.createElement('div');
            tooltipElem.className = 'custom-tooltip';
            tooltipElem.textContent = tooltipText;
            tooltipElem.style.position = 'fixed';
            tooltipElem.style.top = (e.clientY - 40) + 'px';
            tooltipElem.style.left = (e.clientX + 10) + 'px';
            tooltipElem.style.animation = 'fadeInUp 0.3s ease';

            document.body.appendChild(tooltipElem);
            this.tooltipElement = tooltipElem;
        });

        tooltip.addEventListener('mouseleave', function () {
            if (this.tooltipElement) {
                this.tooltipElement.style.animation = 'fadeOutDown 0.3s ease';
                setTimeout(() => {
                    if (this.tooltipElement && this.tooltipElement.parentNode) {
                        this.tooltipElement.parentNode.removeChild(this.tooltipElement);
                    }
                }, 300);
            }
        });

        tooltip.addEventListener('mousemove', function (e) {
            if (this.tooltipElement) {
                this.tooltipElement.style.top = (e.clientY - 40) + 'px';
                this.tooltipElement.style.left = (e.clientX + 10) + 'px';
            }
        });
    });
}

// Enhanced tabs with slide animation
function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            const tabName = this.getAttribute('data-tab');
            const tabContainer = this.closest('.card-body') || document;

            // Deactivate all tabs
            tabContainer.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            tabContainer.querySelectorAll('.tab-content').forEach(c => {
                c.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => {
                    c.classList.remove('active');
                    c.style.animation = '';
                }, 300);
            });

            // Activate current tab
            this.classList.add('active');
            setTimeout(() => {
                const targetTab = tabContainer.querySelector('#' + tabName);
                if (targetTab) {
                    targetTab.classList.add('active');
                    targetTab.style.animation = 'fadeInUp 0.5s ease';
                }
            }, 300);
        });
    });
}

// Initialize interactive elements
function initInteractiveElements() {
    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            this.appendChild(ripple);

            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';

            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    // Add loading states to forms
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;

                // Revert after 5 seconds (fallback)
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 5000);
            }
        });
    });
}

// Scroll progress indicator
function initScrollProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: var(--accent);
        z-index: 1001;
        transition: width 0.1s ease;
    `;
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
        const winHeight = window.innerHeight;
        const docHeight = document.documentElement.scrollHeight;
        const scrollTop = window.pageYOffset;
        const scrollPercent = (scrollTop / (docHeight - winHeight)) * 100;
        progressBar.style.width = scrollPercent + '%';
    });
}

// Basic analytics (optional)
function initAnalytics() {
    // Track page views
    console.log('Page viewed:', window.location.pathname);

    // Track outbound links
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (link && link.href && !link.href.includes(window.location.hostname)) {
            console.log('Outbound link clicked:', link.href);
        }
    });
}

// Enhanced Utility functions
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function () {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Enhanced File handling utilities
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(e);
        reader.readAsText(file);
    });
}

function downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Enhanced Animations
const enhancedStyles = document.createElement('style');
enhancedStyles.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeOutDown {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(20px); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes progressShrink {
        from { width: 100%; }
        to { width: 0%; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple 0.6s linear;
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .fade-in {
        animation: fadeIn 1s ease;
    }
    
    .custom-tooltip {
        background: var(--card);
        color: var(--text);
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid var(--border);
        z-index: 10000;
        pointer-events: none;
        max-width: 200px;
        word-wrap: break-word;
    }
    
    .btn:hover {
        animation: pulse 0.5s ease;
    }
    
    .card {
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Smooth transitions for theme changes */
    * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    
    /* Enhanced scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent);
        border-radius: 6px;
        border: 3px solid var(--bg);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-dark);
    }
`;

document.head.appendChild(enhancedStyles);

// Register CSS properties
if (CSS && CSS.registerProperty) {
    try {
        CSS.registerProperty({
            name: '--gradient-angle',
            syntax: '<angle>',
            initialValue: '0deg',
            inherits: false
        });
    } catch (e) {
        console.log('CSS.registerProperty not supported');
    }
}

// Add floating particles animation for background
function createFloatingParticles() {
    const container = document.querySelector('main');
    if (!container) return;

    for (let i = 0; i < 15; i++) {
        const particle = document.createElement('div');
        particle.className = 'floating-particle';
        particle.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--accent);
            border-radius: 50%;
            opacity: 0.3;
            top: ${Math.random() * 100}%;
            left: ${Math.random() * 100}%;
            animation: float ${Math.random() * 10 + 10}s infinite ease-in-out;
            animation-delay: ${Math.random() * 5}s;
            z-index: -1;
            pointer-events: none;
        `;
        container.appendChild(particle);
    }
}

// Initialize floating particles
createFloatingParticles();

// Add particle animation
const particleStyle = document.createElement('style');
particleStyle.textContent = `
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.3; }
        50% { transform: translateY(-20px) rotate(180deg); opacity: 0.6; }
    }
    
    .floating-particle {
        z-index: -1;
        pointer-events: none;
    }
    
    @media (max-width: 768px) {
        .floating-particle {
            display: none;
        }
    }
`;
document.head.appendChild(particleStyle);

// Performance monitoring
const perf = {
    start: performance.now(),
    init: performance.now()
};

window.addEventListener('load', () => {
    perf.load = performance.now();
    console.log(`Page loaded in ${(perf.load - perf.start).toFixed(2)}ms`);
});

// Error handling
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred', 'error');
});

// Service Worker registration (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(() => console.log('Service Worker registered'))
            .catch(err => console.log('Service Worker registration failed:', err));
    });
}

function fixPlotAlignment() {
    setTimeout(function () {
        document.querySelectorAll('.plotly').forEach(function (plot) {
            plot.style.margin = '0 auto';
            plot.style.display = 'block';
            plot.style.float = 'none';
        });


        document.querySelectorAll('.main-svg').forEach(function (svg) {
            svg.style.margin = '0 auto';
            svg.style.display = 'block';
        });
    }, 1000);
}

document.addEventListener('DOMContentLoaded', fixPlotAlignment);
window.addEventListener('resize', fixPlotAlignment);

document.addEventListener('plotly_afterplot', fixPlotAlignment);

