"""Varied, friendly Course Answer text when retrieval returns no chunks (greetings, vague input)."""

from __future__ import annotations

import random
import re
from typing import Literal

NoMatchKind = Literal["greeting", "short_ack", "off_topic"]

# Single-line greetings / small talk (no course vocabulary expected)
_GREETING_LINE = re.compile(
    r"^("
    r"hi|hello|hey|hi there|hello there|"
    r"good\s+(morning|afternoon|evening)|"
    r"howdy|greetings|"
    r"what(?:'s| is)\s+up|whats up|wassup|sup\b|"
    r"yo\b"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

_SHORT_ACK_LINE = re.compile(
    r"^("
    r"ok|okay|k\b|"
    r"thanks|thank you|thx|ty|tysm|"
    r"got it|makes sense|cool|nice|great|"
    r"bye|goodbye|see ya|cya"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)


def classify_no_match_query(text: str) -> NoMatchKind:
    """
    Classify user text when lexical retrieval found no chunks.

    ``greeting`` — hello/hi-style openers.
    ``short_ack`` — brief acknowledgements / thanks.
    ``off_topic`` — everything else (gibberish, unrelated questions, etc.).
    """
    raw = (text or "").strip()
    if not raw:
        return "off_topic"
    if _GREETING_LINE.match(raw):
        return "greeting"
    if _SHORT_ACK_LINE.match(raw):
        return "short_ack"
    return "off_topic"


def varied_no_chunk_course_answer(kind: NoMatchKind) -> str:
    """
    Return a **Course Answer:** block with rotating copy so repeat visits do not feel identical.

    Not LLM-generated — deterministic pool + :func:`random.choice` per request.
    """
    pools: dict[NoMatchKind, list[str]] = {
        "greeting": [
            "Hey — I'm here for **LING 487**, and I answer from the lecture chunks we have indexed (not the open web). "
            "If you tell me what you're working on, I can define a term, walk through an idea step by step, or compare two things from class — for example asking *what softmax does* or *how attention differs from a recurrent layer* usually pulls the right material. "
            "You can also ask for something like a recap of a specific lecture if you name the lecture number.",
            "Hi! Good to see you. I'm grounded in your course materials, so the more your question uses vocabulary from the slides, the better I can retrieve a solid answer. "
            "That might look like \"explain backprop in plain language,\" \"compare MFCCs and the raw spectrum,\" or \"what's the role of layer norm in a transformer block?\" — each of those gives me something concrete to match against the corpus.",
            "Hello. Whenever you're ready, send a real question from the course and I'll tie the explanation to the uploaded notes. "
            "Questions that work well often sound like: *Why do we use softmax for outputs?* or *How does a CNN differ from a fully connected net for images?* — short greetings are fine, but a technical hook is what unlocks a full answer.",
            "Hey there. Think of me as a tutor that only \"reads\" what was loaded for this class: definitions, derivations, and examples that appear in those chunks. "
            "So if you ask something like \"walk me through the bias–variance tradeoff\" or \"summarize the main ideas from lecture 12,\" I can respond in one coherent explanation with examples drawn from the materials — that's the sweet spot.",
            "Hi — happy to help with LING 487 content. I don't have a live web browse; I match your question to lecture text. "
            "For a strong answer, combine *what you want* with a course concept: e.g. \"give an intuition for autoregressive language modeling\" or \"show how dropout acts as regularization\" — I'll weave the explanation and examples together from the matching sections.\n\n"
            "If you just want ideas, you could start from topics like **attention**, **convolution**, **MFCCs**, or **gradient descent** — but a full sentence still works best.",
            "Good to meet you. Ask me anything that's actually in the syllabus materials: I can clarify jargon, connect two ideas, or unpack a slide-heavy topic in plain language. "
            "For instance, *What is positional encoding for?* or *How do we go from logits to a probability distribution?* are the kind of prompts where I can give you a single, structured explanation instead of a generic pointer.",
        ],
        "short_ack": [
            "Sounds good — whenever you want to dive in, send a question that mentions a topic or term from lecture (something like \"explain softmax\" or \"difference between CNN and fully connected layers\"). "
            "I'll answer in one place with the explanation and examples pulled together from the course chunks, which works better than a vague follow-up.",
            "Got it. Next message, try one concrete ask: e.g. a definition (*what is …?*), a comparison (*X vs Y*), or *can you summarize lecture N?* — that gives retrieval something to latch onto so you get a real explanation, not just steering text.",
            "No problem. I'm here when you need me; just phrase the next turn like you're talking to a TA who only has your slides — name the concept (e.g. **transformers**, **MFCCs**, **regularization**) and what you want out of it (intuition, steps, tradeoffs).",
            "Thanks — ping me with a technical question when you're ready. The answers I give are built from the indexed lectures, so something like \"walk me through backprop for one layer\" or \"why use residual connections?\" lands much better than a one-word prompt.",
            "Cool. Take your time; when you come back, a question that embeds an example works great — for instance \"give an example of when we'd use dynamic programming in NLP\" ties the *topic* and the *kind of answer* together in one go.",
            "Appreciate it. Fire away with your next question using course vocabulary and I'll respond with a unified explanation (definition + intuition + tie-in to lecture) instead of scattered bullet points — that's how this tutor is set up to read best.",
        ],
        "off_topic": [
            "I couldn't match that to anything in the **LING 487** materials we have loaded — sometimes that's wording, sometimes the topic just isn't in the corpus. "
            "If you rephrase around a concept from class, I can usually answer in one flowing explanation with examples from the notes. For instance, try *What is backpropagation doing, intuitively?* or *How does self-attention combine information across positions?* — both give me a clear hook.",
            "Nothing in the indexed lecture chunks lined up with that query. That's often fixed by naming a term from the slides (**softmax**, **spectrogram**, **layer norm**, …) or asking a compare/define question in full sentences. "
            "Example: instead of a short slang line, something like \"explain how MFCCs relate to the mel filterbank\" tells me both *what* and *how deep* to go.",
            "I'm not finding course text for this one yet. I work by lexical match to your uploaded sections, so a question that sounds like how you'd ask a TA — *Can you explain dropout?* or *What's the difference between training loss and validation loss?* — tends to retrieve chunks and lets me give you explanation plus examples in one answer.",
            "That didn't connect to the stored materials. Could be off-topic phrasing, a typo, or a question outside what was imported. "
            "Try again with a concrete LING 487 anchor: e.g. \"describe the forward pass for a single attention head\" or \"summarize the main takeaway from lecture 10\" — I'll keep the response as one narrative with examples drawn from whatever hits.",
            "Retrieval came up empty, so I don't have a grounded paragraph to give you yet. "
            "The fix is usually to fold in vocabulary from lecture and ask for a specific outcome — *intuition for diffusion models*, *when we'd use a CNN vs an MLP*, *what bias vs variance means for model complexity* — so the answer can be structured as explanation-with-examples instead of disconnected bullets.\n\n"
            "If you're not sure where to start, pick any heading you remember from a recent lecture and ask \"what does that mean?\" in your own words.",
            "I don't have a matching chunk for that. I'm not browsing the web; I'm matching your words to lecture content. "
            "Reframe with a term or scenario from class — for example asking *How does stochastic gradient descent differ from full-batch?* or *Why might we use MFCC features instead of raw waveforms?* — and I'll respond with a single coherent write-up tied to the materials.",
        ],
    }
    body = random.choice(pools[kind])
    return f"Course Answer:\n{body}"
