import React, { useEffect, useMemo, useRef, useState } from "react";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function DocTalkLogo() {
  return (
    <div className="logo-wrap" aria-hidden="true">
      <div className="logo-icon">
        <span className="logo-sheet" />
        <span className="logo-chat" />
      </div>
      <span className="logo-text">DocTalk</span>
    </div>
  );
}

function App() {
  const [view, setView] = useState("landing");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [chatItems, setChatItems] = useState([]);
  const [error, setError] = useState("");
  const [scrollProgress, setScrollProgress] = useState(0);
  const chatEndRef = useRef(null);

  const canUpload = useMemo(
    () => selectedFile && !uploading && !uploaded,
    [selectedFile, uploading, uploaded]
  );

  useEffect(() => {
    function onScroll() {
      const top = window.scrollY || 0;
      const maxScrollable = Math.max(document.body.scrollHeight - window.innerHeight, 1);
      const progress = Math.min(top / maxScrollable, 1);
      setScrollProgress(progress);
    }

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatItems]);

  async function handleUpload() {
    if (!selectedFile || uploading || uploaded) {
      return;
    }

    setError("");
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${BASE_URL}/upload`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        let message = "Upload failed. Check backend and try again.";
        try {
          const data = await res.json();
          message = data.detail || message;
        } catch {
          // Keep the generic message if the backend closed before returning JSON.
        }
        throw new Error(message);
      }

      await res.json();
      setUploaded(true);
      setView("chat");
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleAskQuestion() {
    if (!query.trim() || asking) {
      return;
    }

    setError("");
    setAsking(true);
    const question = query.trim();
    setQuery("");
    const messageId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

    setChatItems((prev) => [
      ...prev,
      {
        id: messageId,
        question,
        answer: "Thinking...",
        citations: [],
        loading: true
      }
    ]);

    try {
      const res = await fetch(`${BASE_URL}/chat?query=${encodeURIComponent(question)}`, {
        method: "POST"
      });

      if (!res.ok) {
        throw new Error("Could not get response from backend.");
      }

      const data = await res.json();
      setChatItems((prev) => [
        ...prev.map((item) =>
          item.id === messageId
            ? {
                ...item,
                answer: data.answer || "No answer returned.",
                citations: Array.isArray(data.citations) ? data.citations : [],
                loading: false
              }
            : item
        )
      ]);
    } catch (err) {
      setError(err.message || "Something went wrong while asking.");
      setChatItems((prev) => [
        ...prev.map((item) =>
          item.id === messageId
            ? {
                ...item,
                answer: "I could not fetch a response. Please make sure backend is running.",
                citations: [],
                loading: false
              }
            : item
        )
      ]);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="bg-glow glow-a" />
      <div className="bg-glow glow-b" />

      <header className="navbar glass">
        <DocTalkLogo />
        <nav className="nav-actions">
          <button
            className={`chip-btn ${view === "landing" ? "chip-active" : ""}`}
            onClick={() => setView("landing")}
            type="button"
          >
            Home
          </button>
          <button
            className={`chip-btn ${view === "chat" ? "chip-active" : ""}`}
            onClick={() => setView(uploaded ? "chat" : "landing")}
            type="button"
          >
            Chat
          </button>
        </nav>
      </header>

      <main className="content">
        {view === "landing" && (
          <section
            className="card glass card-animate hero-card"
            style={{ "--scroll-progress": scrollProgress }}
          >
            <div className="hero-noise" />
            <p className="hero-shadow-text">DocTalk</p>
            <div className="spark-orb" />

            <div className="hero-top">
              <p className="eyebrow">Document-Constrained Conversational Agent</p>
              <h1>Powerful document conversations, strict source grounding.</h1>
              <p className="subtitle">
                Upload one document and ask questions. Every response is tuned for citation-backed,
                document-only answers with reliable refusal for out-of-scope prompts.
              </p>
            </div>

            <div className="hero-layout">
              <div className="upload-box">
                <p className="upload-title">Start with your document</p>
                <label className="file-input-wrap">
                  <input
                    type="file"
                    accept=".pdf,.txt,.csv,application/pdf,text/plain,text/csv"
                    onChange={(event) => {
                      const file = event.target.files?.[0] || null;
                      setSelectedFile(file);
                      setUploaded(false);
                    }}
                  />
                  <span>{selectedFile ? selectedFile.name : "Choose PDF, TXT, or CSV file"}</span>
                </label>

                <button className="primary-btn" type="button" disabled={!canUpload} onClick={handleUpload}>
                  {uploading ? "Uploading..." : uploaded ? "Uploaded" : "Upload"}
                </button>
              </div>

              <div className="hero-feature-stack">
                <div className="feature-card">
                  <h3>Source-bound responses</h3>
                  <p>Answers are generated only from retrieved chunks of your uploaded document.</p>
                </div>
                <div className="feature-card">
                  <h3>Out-of-scope refusal</h3>
                  <p>If the document does not support a query, DocTalk declines cleanly.</p>
                </div>
                <div className="feature-card">
                  <h3>Citation first</h3>
                  <p>Every answer can include page references for quick verification.</p>
                </div>
              </div>
            </div>
          </section>
        )}

        {view === "chat" && (
          <section className="card glass card-animate chat-card">
            <h2>DocTalk Chat</h2>
            <p className="subtitle">Ask questions about your uploaded document and review cited responses.</p>

            <div className="chat-stream">
              {chatItems.length === 0 && (
                <div className="chat-empty">
                  <p>Ask your first question about the uploaded document.</p>
                </div>
              )}

              {chatItems.map((item) => (
                <article className="chat-item" key={item.id}>
                  <div className="chat-q bubble user-bubble">
                    <span>You</span>
                    <p>{item.question}</p>
                  </div>
                  <div className="chat-a bubble bot-bubble">
                    <span>DocTalk</span>
                    <p>{item.answer}</p>
                    {!item.loading && (
                      <small>
                        Citations: {item.citations.length > 0 ? item.citations.join(", ") : "None provided"}
                      </small>
                    )}
                  </div>
                </article>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div className="ask-box">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask a question about your document..."
              />
              <button
                className="primary-btn"
                type="button"
                disabled={asking || !query.trim()}
                onClick={handleAskQuestion}
              >
                {asking ? "Thinking..." : "Ask"}
              </button>
            </div>
          </section>
        )}

        {error && <p className="error-text">{error}</p>}
      </main>
    </div>
  );
}

export default App;
