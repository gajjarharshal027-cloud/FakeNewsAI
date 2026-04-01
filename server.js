require('dotenv').config();
const express = require('express');
const fetch = require('node-fetch');
const path = require('path');
const net = require('net');

const app = express();
const NEWS_API_KEY = process.env.NEWS_API_KEY;

app.use(express.json());
app.use(express.static(__dirname));

// ── OpenEnv required state ────────────────────────────────────────────
let envState = {
  headline: '',
  verdict: null,
  confidence: null,
  articles: [],
  step_count: 0,
  done: false
};

// ── OpenEnv: GET /health ──────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', environment: 'FakeNewsAI', version: '1.0.0' });
});

// ── OpenEnv: POST /reset ──────────────────────────────────────────────
app.post('/reset', (req, res) => {
  envState = {
    headline: '',
    verdict: null,
    confidence: null,
    articles: [],
    step_count: 0,
    done: false
  };
  res.status(200).json({
    observation: envState,
    info: { message: 'Environment reset successfully' }
  });
});

// ── OpenEnv: POST /step ───────────────────────────────────────────────
app.post('/step', async (req, res) => {
  const { action } = req.body;
  envState.headline = action?.headline || '';
  envState.step_count += 1;

  res.status(200).json({
    observation: envState,
    reward: 0,
    done: false,
    info: { step: envState.step_count }
  });
});

// ── OpenEnv: GET /state ───────────────────────────────────────────────
app.get('/state', (req, res) => {
  res.status(200).json(envState);
});

// ── Serve frontend ────────────────────────────────────────────────────
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ── Groq AI fallback ──────────────────────────────────────────────────
async function verifyWithGroq(headline, groqKey) {
  const prompt = `You are a professional fact-checker. Analyze this headline and determine if it is TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE.\n\nHeadline: "${headline}"\n\nRespond ONLY with a JSON object (no markdown) in this format:\n{"verdict":"TRUE"|"FALSE"|"PARTIAL"|"UNVERIFIABLE","confidence":"High"|"Medium"|"Low","explanation":"2-3 sentences","keyClaim":"the main claim"}`;

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${groqKey}` },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      max_tokens: 500,
      messages: [
        { role: 'system', content: 'You are a fact-checker. Respond with valid JSON only, no markdown.' },
        { role: 'user', content: prompt }
      ]
    })
  });

  if (!response.ok) { const e = await response.json(); throw new Error(e.error?.message || 'Groq error'); }
  const data = await response.json();
  const clean = data.choices[0].message.content.trim().replace(/```json|```/g, '').trim();
  return JSON.parse(clean);
}

// ── Main verify endpoint ──────────────────────────────────────────────
app.get('/api/verify', async (req, res) => {
  const { headline, groqKey } = req.query;
  if (!headline) return res.status(400).json({ error: 'Missing headline' });

  try {
    const query = encodeURIComponent(headline.substring(0, 100));
    const newsUrl = `https://newsapi.org/v2/everything?q=${query}&pageSize=5&sortBy=relevancy&apiKey=${NEWS_API_KEY}`;
    const newsResponse = await fetch(newsUrl);
    const newsData = await newsResponse.json();
    const articles = newsData.articles || [];

    if (articles.length > 0) {
      const verdict = articles.length >= 3 ? 'TRUE' : 'PARTIAL';
      const confidence = articles.length >= 3 ? 'High' : 'Medium';
      return res.json({
        source: 'newsapi', verdict, confidence,
        totalResults: newsData.totalResults,
        articles: articles.map(a => ({ title: a.title, source: a.source?.name, url: a.url, publishedAt: a.publishedAt }))
      });
    }

    if (!groqKey) return res.json({ source: 'newsapi', verdict: 'UNVERIFIABLE', confidence: 'Low', totalResults: 0, articles: [] });

    const result = await verifyWithGroq(headline, groqKey);
    return res.json({ source: 'groq', ...result, totalResults: 0, articles: [] });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── Auto free port ────────────────────────────────────────────────────
function findFreePort(start) {
  return new Promise(resolve => {
    const s = net.createServer();
    s.listen(start, () => { const p = s.address().port; s.close(() => resolve(p)); });
    s.on('error', () => resolve(findFreePort(start + 1)));
  });
}

findFreePort(process.env.PORT || 3000).then(port => {
  app.listen(port, () => {
    console.log('');
    console.log('  ✅ TruthLens is running!');
    console.log(`  👉 http://localhost:${port}`);
    console.log('  📡 OpenEnv endpoints: /health  /reset  /step  /state');
    console.log('');
  });
});
