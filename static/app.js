// RL-Enhanced Self-Healing CRAG Frontend Controller

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const fileDropzone = document.getElementById('fileDropzone');
  const fileInput = document.getElementById('fileInput');
  const uploadPreview = document.getElementById('uploadSelectedFileName');
  const btnUploadSubmit = document.getElementById('btnUploadSubmit');
  const manualDocTitle = document.getElementById('manualDocTitle');
  const manualDocContent = document.getElementById('manualDocContent');
  const btnTextSubmit = document.getElementById('btnTextSubmit');
  const btnLoadSampleSpec = document.getElementById('btnLoadSampleSpec');
  const docListContainer = document.getElementById('docListContainer');
  const docCountPill = document.getElementById('docCountPill');
  const btnResetAll = document.getElementById('btnResetAll');
  const chromaStatusLine = document.getElementById('chromaStatusLine');
  const chromaStatusBadge = document.getElementById('chromaStatusBadge');
  const jsonlCountPill = document.getElementById('jsonlCountPill');
  const jsonlFileList = document.getElementById('jsonlFileList');
  const jsonlViewer = document.getElementById('jsonlViewer');
  const jsonlViewerLabel = document.getElementById('jsonlViewerLabel');
  const btnCopyJsonl = document.getElementById('btnCopyJsonl');

  const queryInput = document.getElementById('queryInput');
  const actionOverrideSelect = document.getElementById('actionOverrideSelect');
  const btnExecuteQuery = document.getElementById('btnExecuteQuery');
  const presetBtns = document.querySelectorAll('.preset-btn');

  // Visualizer Nodes
  const stepNodeMMR = document.getElementById('stepNodeMMR');
  const stepNodeRerank = document.getElementById('stepNodeRerank');
  const stepNodeState = document.getElementById('stepNodeState');
  const stepNodeBandit = document.getElementById('stepNodeBandit');
  const stepNodeAction = document.getElementById('stepNodeAction');
  const stepNodeReward = document.getElementById('stepNodeReward');
  const activeActionBadge = document.getElementById('activeActionBadge');

  const mmrMetric = document.getElementById('mmrMetric');
  const rerankMetric = document.getElementById('rerankMetric');
  const stateMetric = document.getElementById('stateMetric');
  const banditChosenMetric = document.getElementById('banditChosenMetric');
  const branchSub = document.getElementById('branchSub');
  const branchMetric = document.getElementById('branchMetric');
  const rewardMetric = document.getElementById('rewardMetric');

  // Response & Telemetry
  const responseOutput = document.getElementById('responseOutput');
  const rewardStrip = document.getElementById('rewardStrip');
  const stripJudgeScore = document.getElementById('stripJudgeScore');
  const stripPenalty = document.getElementById('stripPenalty');
  const stripNetReward = document.getElementById('stripNetReward');

  const retrievedNodesList = document.getElementById('retrievedNodesList');
  const branchNoticeArea = document.getElementById('branchNoticeArea');
  const mathArmsBreakdown = document.getElementById('mathArmsBreakdown');
  const rawJsonTrace = document.getElementById('rawJsonTrace');
  const btnCopyJson = document.getElementById('btnCopyJson');

  // Bandit Arms Telemetry
  const banditStepsCount = document.getElementById('banditStepsCount');
  const a0Count = document.getElementById('a0Count');
  const a0Avg = document.getElementById('a0Avg');
  const a0Bar = document.getElementById('a0Bar');
  const a1Count = document.getElementById('a1Count');
  const a1Avg = document.getElementById('a1Avg');
  const a1Bar = document.getElementById('a1Bar');
  const a2Count = document.getElementById('a2Count');
  const a2Avg = document.getElementById('a2Avg');
  const a2Bar = document.getElementById('a2Bar');

  let selectedUploadFile = null;
  let lastTelemetryJson = null;
  let lastJsonlText = '';

  // 1. Tab Switching
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const parent = btn.closest('.card') || btn.closest('.results-grid') || document;
      const targetTabId = btn.dataset.tab;

      parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(targetTabId);
      if (targetContent) targetContent.classList.add('active');
    });
  });

  // 2. File Upload Dropzone
  fileDropzone.addEventListener('click', () => fileInput.click());
  fileDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileDropzone.classList.add('dragover');
  });
  fileDropzone.addEventListener('dragleave', () => fileDropzone.classList.remove('dragover'));
  fileDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    fileDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  function handleFileSelection(file) {
    selectedUploadFile = file;
    uploadPreview.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    uploadPreview.classList.remove('hidden');
    btnUploadSubmit.disabled = false;
  }

  function showIngestResult(data) {
    const embedded = data.embedded ? 'yes' : 'no';
    const msg =
      `Ingested ${data.num_chunks} nodes → embedded (${embedded}, dim=${data.embedding_dim || 384}) ` +
      `→ ChromaDB count=${data.chroma_count}. JSONL: ${data.jsonl_path || 'n/a'}`;
    alert(msg);
    if (Array.isArray(data.jsonl_lines) && data.jsonl_lines.length) {
      renderJsonlRecords(data.jsonl_lines, data.doc_name || data.jsonl_path || 'latest ingest');
    }
  }

  function renderJsonlRecords(records, label) {
    lastJsonlText = records.map(r => JSON.stringify(r)).join('\n');
    jsonlViewer.textContent = records.map(r => JSON.stringify(r, null, 2)).join('\n\n');
    if (jsonlViewerLabel) {
      jsonlViewerLabel.textContent = `${label} (${records.length} lines)`;
    }
  }

  // 3. Ingestion Actions
  btnUploadSubmit.addEventListener('click', async () => {
    if (!selectedUploadFile) return;
    btnUploadSubmit.disabled = true;
    btnUploadSubmit.textContent = 'Parsing → Embed → Chroma…';

    const formData = new FormData();
    formData.append('file', selectedUploadFile);

    try {
      const res = await fetch('/api/ingest/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        showIngestResult(data);
        selectedUploadFile = null;
        fileInput.value = '';
        uploadPreview.classList.add('hidden');
        await refreshCatalog();
      } else {
        alert(`Ingestion failed: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      btnUploadSubmit.disabled = false;
      btnUploadSubmit.textContent = 'Parse & embed';
    }
  });

  btnTextSubmit.addEventListener('click', async () => {
    const text = manualDocContent.value.trim();
    const title = manualDocTitle.value.trim() || 'Manual_Spec.txt';
    if (!text) {
      alert('Please enter text content.');
      return;
    }

    btnTextSubmit.disabled = true;
    btnTextSubmit.textContent = 'Embed & Index…';

    try {
      const res = await fetch('/api/ingest/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, title })
      });
      const data = await res.json();
      if (res.ok) {
        showIngestResult(data);
        manualDocContent.value = '';
        manualDocTitle.value = '';
        await refreshCatalog();
      } else {
        alert(`Error: ${data.detail || 'Failed to index'}`);
      }
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      btnTextSubmit.disabled = false;
      btnTextSubmit.textContent = 'Index nodes';
    }
  });

  btnLoadSampleSpec.addEventListener('click', async () => {
    btnLoadSampleSpec.disabled = true;
    btnLoadSampleSpec.textContent = 'Loading Helios Spec…';
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      await refreshCatalog();
      await updateBanditStats();
      // Load the seeded JSONL into the viewer
      const jsonlRes = await fetch('/api/jsonl');
      const jsonlData = await jsonlRes.json();
      if (jsonlData.files && jsonlData.files.length) {
        await loadJsonlFile(jsonlData.files[0].filename);
      }
      alert(data.message || 'Helios Fusion Reactor Docling specification loaded, embedded, and stored in ChromaDB.');
    } catch (err) {
      alert(`Error loading sample: ${err.message}`);
    } finally {
      btnLoadSampleSpec.disabled = false;
      btnLoadSampleSpec.textContent = 'Load sample specification';
    }
  });

  async function refreshCatalog() {
    await loadDocumentList();
    await loadJsonlCatalog();
  }

  // 4. Load Indexed Documents Catalog
  async function loadDocumentList() {
    try {
      const res = await fetch('/api/documents');
      const data = await res.json();
      if (data.documents && data.documents.length) {
        docCountPill.textContent = data.total;
        docListContainer.innerHTML = data.documents.map(doc => `
          <div class="doc-item">
            <span class="doc-item-title" title="${doc.name}">${doc.name}</span>
            <span class="doc-item-count">${doc.chunk_count} nodes</span>
          </div>
        `).join('');
      } else {
        docCountPill.textContent = '0';
        docListContainer.innerHTML = '<div class="empty-hint">No documents indexed yet.</div>';
      }

      const chroma = data.chroma || {};
      if (chromaStatusLine) {
        chromaStatusLine.textContent =
          `ChromaDB: ${chroma.chunk_count || 0} embedded chunks · ${chroma.embedding_model || 'MiniLM'} · ready=${chroma.ready_for_retrieval ? 'yes' : 'no'}`;
      }
      if (chromaStatusBadge) {
        const label = chromaStatusBadge.querySelector('span:last-child');
        if (label) {
          label.textContent = chroma.ready_for_retrieval
            ? `Chroma · ${chroma.chunk_count || 0}`
            : 'Chroma · empty';
        }
      }
    } catch (err) {
      console.warn('Failed to load documents catalog:', err);
    }
  }

  async function loadJsonlCatalog() {
    try {
      const res = await fetch('/api/jsonl');
      const data = await res.json();
      const files = data.files || [];
      jsonlCountPill.textContent = files.length;
      if (!files.length) {
        jsonlFileList.innerHTML = '<div class="empty-hint">JSONL appears here after ingest.</div>';
        return;
      }
      jsonlFileList.innerHTML = files.map((f, idx) => `
        <div class="jsonl-file-item ${idx === 0 ? 'active' : ''}" data-filename="${f.filename}">
          <span class="jsonl-name" title="${f.filename}">${f.filename}</span>
          <span class="doc-item-count">${f.num_lines} lines</span>
        </div>
      `).join('');

      jsonlFileList.querySelectorAll('.jsonl-file-item').forEach(el => {
        el.addEventListener('click', () => {
          jsonlFileList.querySelectorAll('.jsonl-file-item').forEach(x => x.classList.remove('active'));
          el.classList.add('active');
          loadJsonlFile(el.dataset.filename);
        });
      });

      // Auto-show newest file if viewer is still placeholder
      if (!lastJsonlText) {
        await loadJsonlFile(files[0].filename);
      }
    } catch (err) {
      console.warn('Failed to load JSONL catalog:', err);
    }
  }

  async function loadJsonlFile(filename) {
    try {
      const res = await fetch(`/api/jsonl/${encodeURIComponent(filename)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load JSONL');
      renderJsonlRecords(data.records || [], filename);
    } catch (err) {
      jsonlViewer.textContent = `Error loading JSONL: ${err.message}`;
    }
  }

  if (btnCopyJsonl) {
    btnCopyJsonl.addEventListener('click', () => {
      if (!lastJsonlText) return;
      navigator.clipboard.writeText(lastJsonlText);
      btnCopyJsonl.textContent = 'Copied';
      setTimeout(() => { btnCopyJsonl.textContent = 'Copy'; }, 2000);
    });
  }

  // 5. Preset Scenario Buttons
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.dataset.query;
      actionOverrideSelect.value = btn.dataset.action;
      executeCRAGPipeline();
    });
  });

  // 6. Execute CRAG Query
  btnExecuteQuery.addEventListener('click', executeCRAGPipeline);

  async function executeCRAGPipeline() {
    const query = queryInput.value.trim();
    if (!query) {
      alert('Please enter a query.');
      return;
    }

    const forcedActionVal = actionOverrideSelect.value;
    const forced_action = forcedActionVal === 'auto' ? null : parseInt(forcedActionVal, 10);

    btnExecuteQuery.disabled = true;
    btnExecuteQuery.innerHTML = '<span class="pulse-dot"></span><span>Running…</span>';
    responseOutput.innerHTML = '<div class="empty-response-placeholder"><span class="pulse-dot"></span> Retrieving and ranking…</div>';
    rewardStrip.classList.add('hidden');
    resetStepNodes();

    try {
      activateStepNode(stepNodeMMR);

      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, forced_action })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Query execution failed.');
      }

      lastTelemetryJson = data;

      animatePipelineExecution(data);
      renderSynthesizedResponse(data);
      renderRetrievedNodes(data);
      renderBanditMath(data);
      renderRawTrace(data);
      await updateBanditStats();

    } catch (err) {
      responseOutput.innerHTML = `<div style="color: var(--color-danger); padding: 12px;">Execution Error: ${err.message}</div>`;
    } finally {
      btnExecuteQuery.disabled = false;
      btnExecuteQuery.innerHTML = '<span>Run</span>';
    }
  }

  function resetStepNodes() {
    [stepNodeMMR, stepNodeRerank, stepNodeState, stepNodeBandit, stepNodeAction, stepNodeReward].forEach(node => {
      node.classList.remove('active-step');
    });
    activeActionBadge.className = 'action-tag idle';
    activeActionBadge.textContent = 'Processing...';
  }

  function activateStepNode(node) {
    node.classList.add('active-step');
  }

  function animatePipelineExecution(data) {
    activateStepNode(stepNodeMMR);
    mmrMetric.textContent = `${data.retrieval_metrics.num_candidates_mmr} cands`;

    setTimeout(() => {
      activateStepNode(stepNodeRerank);
      rerankMetric.textContent = `${data.retrieval_metrics.num_reranked} ranked`;
    }, 200);

    setTimeout(() => {
      activateStepNode(stepNodeState);
      stateMetric.textContent = `Avg:${data.state_vector_summary.score_avg} | Max:${data.state_vector_summary.score_max}`;
    }, 400);

    setTimeout(() => {
      activateStepNode(stepNodeBandit);
      banditChosenMetric.textContent = `Arm A${data.action} selected`;
      activeActionBadge.className = `action-tag a${data.action}`;
      activeActionBadge.textContent = `Action ${data.action}: ${data.action_name}`;
    }, 600);

    setTimeout(() => {
      activateStepNode(stepNodeAction);
      if (data.action === 0) {
        branchSub.textContent = 'Direct Generation';
        branchMetric.textContent = `${data.retrieved_docs.length} local chunks`;
      } else if (data.action === 1) {
        branchSub.textContent = 'Query Rewritten';
        branchMetric.textContent = `Re-retrieved & merged`;
      } else if (data.action === 2) {
        branchSub.textContent = 'DuckDuckGo Fallback';
        branchMetric.textContent = `${data.external_search_results ? data.external_search_results.length : 0} web hits`;
      }
    }, 800);

    setTimeout(() => {
      activateStepNode(stepNodeReward);
      rewardMetric.textContent = `R_t = ${data.net_reward.toFixed(2)}`;
    }, 1000);
  }

  function renderSynthesizedResponse(data) {
    responseOutput.textContent = data.response;

    stripJudgeScore.textContent = data.judge_score.toFixed(2);
    stripPenalty.textContent = data.action_penalty.toFixed(2);
    stripNetReward.textContent = data.net_reward.toFixed(2);
    rewardStrip.classList.remove('hidden');
  }

  function renderRetrievedNodes(data) {
    branchNoticeArea.className = 'branch-notice hidden';
    branchNoticeArea.innerHTML = '';

    if (data.action === 1 && data.rewritten_query) {
      branchNoticeArea.className = 'branch-notice action1';
      branchNoticeArea.innerHTML = `<strong>Action 1 · Rewrite</strong> — expanded to <em>"${data.rewritten_query}"</em>`;
    } else if (data.action === 2 && data.external_search_results && data.external_search_results.length) {
      branchNoticeArea.className = 'branch-notice action2';
      branchNoticeArea.innerHTML = `<strong>Action 2 · External</strong> — ${data.external_search_results.length} web sources`;
    }

    let itemsHtml = '';

    if (data.action === 2 && data.external_search_results) {
      data.external_search_results.forEach((item, idx) => {
        itemsHtml += `
          <div class="node-item-card">
            <div class="node-item-header">
              <span class="node-ref-tag">External #${idx + 1}</span>
              <span class="node-score-badge">Web</span>
            </div>
            <div style="font-weight: 650; margin-bottom: 4px; font-size: 0.84rem;">${item.title}</div>
            <div class="node-text-preview">${item.snippet}</div>
            <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; overflow:hidden; text-overflow:ellipsis;">${item.source}</div>
          </div>
        `;
      });
    }

    if (data.retrieved_docs && data.retrieved_docs.length) {
      data.retrieved_docs.forEach(doc => {
        const ref = doc.metadata ? doc.metadata.node_ref : '#/node';
        const label = doc.metadata ? doc.metadata.label : 'text';
        const score = doc.rerank_score !== undefined ? doc.rerank_score.toFixed(4) : (doc.cosine_sim || 0.0).toFixed(4);
        const payload = doc.payload || {};
        const isMultimodal = payload.image_description || label === 'picture';

        itemsHtml += `
          <div class="node-item-card">
            <div class="node-item-header">
              <span class="node-ref-tag">${ref} · ${label}</span>
              <span class="node-score-badge">${score}</span>
            </div>
            <div class="node-text-preview">${doc.text}</div>
            ${isMultimodal ? `<div class="multimodal-pill">Vision enriched</div>` : ''}
          </div>
        `;
      });
    }

    const mmr = data.retrieval_metrics ? data.retrieval_metrics.num_candidates_mmr : 0;
    if (!itemsHtml && mmr === 0) {
      itemsHtml = '<div class="empty-hint">No chunks in ChromaDB — ingest a document first (JSONL → embed → store).</div>';
    }

    retrievedNodesList.innerHTML = itemsHtml || '<div class="empty-hint">No nodes retrieved.</div>';
  }

  function renderBanditMath(data) {
    const tel = data.bandit_telemetry;
    if (!tel) return;

    const actionNames = {
      0: "Direct Generation",
      1: "Query Expansion & Rewrite",
      2: "External Fallback Search"
    };

    let breakdownHtml = '';
    [0, 1, 2].forEach(a => {
      const isChosen = a === data.action;
      const ucb = tel.arm_scores[a].toFixed(4);
      const mean = tel.arm_mean_payoff[a].toFixed(4);
      const bonus = tel.arm_variance_bonus[a].toFixed(4);

      breakdownHtml += `
        <div class="arm-math-row ${isChosen ? 'chosen' : ''}">
          <div class="arm-math-title">
            <span>Action ${a}: ${actionNames[a]}</span>
            <span style="color: ${isChosen ? 'var(--accent-cyan)' : 'var(--text-secondary)'}; font-family: var(--font-mono);">
              p_{t,${a}} = ${ucb} ${isChosen ? '· chosen' : ''}
            </span>
          </div>
          <div class="arm-math-calc">
            θ̂ᵀ x: <strong>${mean}</strong> (Exploitation) + α √(xᵀ A⁻¹ x): <strong>${bonus}</strong> (Exploration Bonus)
          </div>
        </div>
      `;
    });

    mathArmsBreakdown.innerHTML = breakdownHtml;
  }

  function renderRawTrace(data) {
    rawJsonTrace.textContent = JSON.stringify(data, null, 2);
  }

  btnCopyJson.addEventListener('click', () => {
    if (lastTelemetryJson) {
      navigator.clipboard.writeText(JSON.stringify(lastTelemetryJson, null, 2));
      btnCopyJson.textContent = 'Copied';
      setTimeout(() => { btnCopyJson.textContent = 'Copy'; }, 2000);
    }
  });

  // 7. Update Bandit Stats Telemetry
  async function updateBanditStats() {
    try {
      const res = await fetch('/api/bandit/stats');
      const data = await res.json();
      if (!data || !data.arms) return;

      banditStepsCount.textContent = `${data.total_steps} steps`;

      const arms = data.arms;
      if (arms[0]) {
        a0Count.textContent = `${arms[0].count}`;
        a0Avg.textContent = arms[0].avg_reward.toFixed(2);
        a0Bar.style.width = `${Math.min(100, Math.max(0, arms[0].avg_reward * 100))}%`;
      }
      if (arms[1]) {
        a1Count.textContent = `${arms[1].count}`;
        a1Avg.textContent = arms[1].avg_reward.toFixed(2);
        a1Bar.style.width = `${Math.min(100, Math.max(0, arms[1].avg_reward * 100))}%`;
      }
      if (arms[2]) {
        a2Count.textContent = `${arms[2].count}`;
        a2Avg.textContent = arms[2].avg_reward.toFixed(2);
        a2Bar.style.width = `${Math.min(100, Math.max(0, arms[2].avg_reward * 100))}%`;
      }
    } catch (err) {
      console.warn('Failed to fetch bandit stats:', err);
    }
  }

  // 8. Reset Pipeline
  btnResetAll.addEventListener('click', async () => {
    if (confirm('Reset ChromaDB collections and LinUCB bandit parameters?')) {
      try {
        await fetch('/api/reset', { method: 'POST' });
        lastJsonlText = '';
        await refreshCatalog();
        await updateBanditStats();
        alert('Pipeline, ChromaDB, and LinUCB parameters reset to clean baseline.');
      } catch (err) {
        alert(`Reset error: ${err.message}`);
      }
    }
  });

  // Initial Load
  refreshCatalog();
  updateBanditStats();
});
