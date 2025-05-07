# pip install --upgrade langgraph langchain-anthropic "langchain[anthropic]"
# pip install -U langchain-community tavily-python mcp anthropic python-dotenv

"""Multi‑agent GitHub‑MCP client powered by LangGraph.

Pipeline (when `use_tavily=True`):
    Researcher  ➜  Coder  ➜  Senior  ➜  (Security ║ Performance)  ➜  Tester

• Every agent receives a **CODEBASE OVERVIEW** string that summarizes the repo’s
  main files (first `max_files` paths, first `preview_chars` chars each).
• Tester additionally receives the full project guidelines and must reply either
  **ACCEPT** or **REJECT …**
• If any reviewer or the tester rejects, the flow loops back to the Coder.
• `process_query` returns `(code_blocks, prose)` where `code_blocks` contains the
  concatenated contents of all ``` fenced blocks from the final response.

Run with:
    python github_mcp_client.py
(Ensure Docker is running and the `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`, and
 `TAVILY_API_KEY` environment variables are set in a `.env` file or shell.)
"""

import asyncio
import os
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from collections.abc import Iterable
from dotenv import load_dotenv
load_dotenv()

# ── MCP / IO -----------------------------------------------------------------
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Claude / LangChain‑related ------------------------------------------------
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableParallel
from langchain_community.tools.tavily_search.tool import TavilySearchResults

# ──────────────────────────────────────────────────────────────────────────────
class GitHubMCPClient:
    """Multi‑agent wrapper around GitHub‑MCP server."""

    # ‑‑‑ constructor & connection helpers -----------------------------------
    def __init__(self) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_github_server(self) -> None:
        """Start/attach to the GitHub MCP server (Docker) and list tools."""
        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        resp = await self.session.list_tools()
        print("\nConnected to GitHub MCP server with tools:", [t.name for t in resp.tools])

    # ‑‑‑ static helper --------------------------------------------------------
    @staticmethod
    def _extract_code_and_comments(text: str) -> Tuple[str, str]:
        """Return (code, prose) from a Claude response string."""
        code_blocks = re.findall(r"```(?:[^\n]*\n)?(.*?)```", text, re.S)
        code_text   = "\n\n".join(cb.strip() for cb in code_blocks)
        prose_text  = re.sub(r"```(?:[^\n]*\n)?.*?```", "", text, flags=re.S).strip()
        return code_text, prose_text

    # ‑‑‑ repo + guideline summaries ------------------------------------------
    async def _read_guidelines(self) -> str:
        """Combine README and GUIDELINES.txt into one string."""
        try:
            # ensure we have repo context
            if not hasattr(self, "current_repo"):
                return "[Guidelines unavailable – no repo set]"

            owner = self.current_repo["owner"]
            repo  = self.current_repo["repo"]

            readme = await self.session.call_tool(
                "get_file_contents",
                {"owner": owner, "repo": repo, "path": "README.md"}
            )
            rules = await self.session.call_tool(
                "get_file_contents",
                {"owner": owner, "repo": repo, "path": "guidelines.txt"}
            )
            return f"{readme.content}\n\n---\n\n{rules.content}"
        except Exception:
            return "[Guidelines unavailable]"

    async def _get_repo_summary(self, *, max_files: int = 20, preview_chars: int = 400) -> str:
        """Return a lightweight textual overview of the repo."""
        # ensure we have repo context
        if not hasattr(self, "current_repo"):
            return "[Could not list repository files – no repo set]"
        owner = self.current_repo["owner"]
        repo  = self.current_repo["repo"]

        try:
            # list directory entries at the repository root
            resp = await self.session.call_tool(
                "get_file_contents",
                {"owner": owner, "repo": repo, "path": ""}
            )
            import json
            entries = json.loads(resp.content)
            # extract file paths and limit to max_files
            paths = [e["path"] for e in entries][:max_files]
        except Exception:
            return "[Could not list repository files]"

        snippets: List[str] = []
        for p in paths:
            if any(p.lower().endswith(ext) for ext in (".py", ".md", ".txt")):
                try:
                    file_resp = await self.session.call_tool(
                        "get_file_contents",
                        {"owner": owner, "repo": repo, "path": p}
                    )
                    content = file_resp.content
                    preview = content[:preview_chars]
                    snippets.append(f"### {p}\n{preview}")
                except Exception:
                    continue

        return "\n\n".join(snippets) if snippets else "[No readable files captured]"

    # ── universal flattener -------------------------------------------------
    @staticmethod
    def _flatten_messages(seq) -> List[str]:
        """
        Recursively walk through seq and return a flat list of strings.
        Accepts LangChain Message objects, strings, or nested iterables.
        """
        out: List[str] = []
        for item in seq:
            if isinstance(item, (str, bytes)):
                out.append(item if isinstance(item, str) else item.decode())
            elif hasattr(item, "content"):            # LangChain Message
                out.append(str(item.content))
            elif isinstance(item, Iterable):
                out.extend(GitHubMCPClient._flatten_messages(item))
            else:
                out.append(str(item))
        return out

    # ‑‑‑ core workflow --------------------------------------------------------
    async def process_query(self, query: str, *, use_tavily: bool = False) -> Tuple[str, str]:
        """Run the multi‑agent pipeline and return (code, prose)."""

        # 1️⃣  Collect tool wrappers ------------------------------------------
        resp = await self.session.list_tools()
        wrapped_tools: List[Tool] = []

        def _wrap(t):
            async def _acall(*args, _tool_name=t.name, **kwargs):
                """
                Normalise every calling convention LangChain / Claude produces:
                • positional dict
                • positional JSON string
                • positional "owner/repo" string
                • kwargs with "__arg1"
                • regular kwargs matching the MCP schema
                """
                import json

                # ① convert single positional arg → kwargs
                if args:
                    if len(args) != 1:
                        raise ValueError(f"Unexpected positional args: {args}")
                    raw = args[0]
                    if isinstance(raw, dict):
                        kwargs = raw
                    elif isinstance(raw, str):
                        # try JSON first
                        try:
                            obj = json.loads(raw)
                            if isinstance(obj, dict):
                                kwargs = obj
                            else:
                                raise ValueError
                        except ValueError:
                            # maybe "owner/repo"
                            if "/" in raw:
                                owner, repo = raw.split("/", 1)
                                kwargs = {"owner": owner, "repo": repo}
                            else:
                                raise ValueError(f"Cannot parse positional arg: {raw}")
                    else:
                        raise ValueError(f"Cannot parse positional arg type: {type(raw)}")

                # ② unwrap {"__arg1": ...}
                if "__arg1" in kwargs:
                    return await _acall(kwargs.pop("__arg1"))

                # ③ finally call the real tool
                if {"owner", "repo"} <= kwargs.keys():
                    self.current_repo = {"owner": kwargs["owner"], "repo": kwargs["repo"]}
                res = await self.session.call_tool(_tool_name, kwargs)
                return res.content

            def _scall(*args, **kw):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_acall(*args, **kw))

            wrapped_tools.append(
                Tool(
                    name=t.name,
                    description=t.description,
                    func=_scall,
                    coroutine=_acall,
                )
            )
        for t in resp.tools:
            _wrap(t)
        tools_info = {t.name: t.inputSchema for t in resp.tools}
        #print("[DEBUG] Tools from server:\n", tools_info)
        if use_tavily and os.getenv("TAVILY_API_KEY"):
            wrapped_tools.append(TavilySearchResults(max_results=10, include_answer=True))

        # 2️⃣  Repo + guideline context ---------------------------------------
        guidelines_text = await self._read_guidelines()
        repo_summary    = await self._get_repo_summary()
        print("\n[INFO] Repo summary:\n", repo_summary)
        print("\n[INFO] Project guidelines:\n", guidelines_text)
        base_system_block = (
            "Multi‑agent GitHub MCP session.\n"
            f"CODEBASE OVERVIEW:\n{repo_summary}"
        )        
        # 3️⃣  Agent factory ---------------------------------------------------
        def make_agent(role: str, *, temp: float = 0.7):
            # ① build the system text
            system_block = (
                "You are part of a multi‑agent GitHub development team using MCP tools.\n"
                f"ROLE: {role}\n\nCODEBASE OVERVIEW:\n{repo_summary}"
            )
            if role.startswith("Unit‑Testing Engineer"):
                system_block += (
                    "\n\nPROJECT GUIDELINES:\n" + guidelines_text +
                    "\nReturn exactly 'ACCEPT' if the patch meets *all* requirements; "
                    "otherwise return 'REJECT' followed by a brief reason."
                )

            # ② full ReAct prompt template
            template = f"""{system_block}

            You have access to the following tools:
            {{tools}}

            Use the following format:

            Question: {{input}}
            Thought: you should think about what to do
            Action: the action to take, should be one of [{{tool_names}}]
            Action Input: the input to the action
            Observation: the result of the action
            ...(repeat Thought/Action/Input/Observation as needed)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input

            Begin!

            Question: {{input}}
            Thought:{{agent_scratchpad}}"""

            # prompt = PromptTemplate.from_template(
            #     template,
            #     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            # )
            # prompt = PromptTemplate(
            #     template=template,
            #     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            # )

            # ③ create the agent
            # return create_react_agent(
            #     ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
            #                 temperature=temp),
            #     wrapped_tools,
            #     prompt=prompt,        # <‑‑ use prompt= instead of system_message=
            # )
            llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022",
                        temperature=temp)

            return create_react_agent(llm, wrapped_tools)

        # Agents --------------------------------------------------------------
        if use_tavily:
            researcher_agent = make_agent("Research Engineer – gather external information", temp=0.2)
        coder_agent      = make_agent("Junior Software Engineer")
        senior_agent     = make_agent("Senior Software Engineer", temp=0.3)
        security_agent   = make_agent("Security Reviewer", temp=0.1)
        perf_agent       = make_agent("Performance Reviewer", temp=0.1)
        tester_agent     = make_agent("Unit‑Testing Engineer", temp=0.0)

        # 4️⃣  Parallel reviewers ---------------------------------------------
        reviewers_parallel = RunnableParallel({
            "security": security_agent,
            "performance": perf_agent,
        })

        # 5️⃣  Graph definition ----------------------------------------------
        class DevState(TypedDict):
            messages: List[str]
            review_results: Dict[str, str]
            verdict: Literal["ACCEPT", "REJECT"]
            reject_count: int

        g = StateGraph(DevState)

        # ---- Nodes ------------------------------------------------------------------
        if use_tavily:
            g.add_node("researcher", researcher_agent)

        g.add_node("coder",      coder_agent)
        g.add_node("senior",     senior_agent)
        g.add_node("reviewers",  reviewers_parallel)
        def _msg_to_str(m):
            # LangChain message objects have .content; fall back to str() otherwise
            return getattr(m, "content", str(m))
        # safer merge: tolerate cases where no reviewer output exists
        def _merge(state: DevState) -> DevState:
            reviews = state.get("reviewers", {})          # may be absent
            verdict = (
                "REJECT" if any(r.startswith("REJECT") for r in reviews.values())
                else "ACCEPT"
            )
            new_count = state["reject_count"] + (1 if verdict == "REJECT" else 0)
            return {
                **state,
                "review_results": reviews,
                "verdict": verdict,
                "reject_count": new_count,
                "messages": state["messages"] + GitHubMCPClient._flatten_messages(reviews.values()),
            }

        g.add_node("merge", _merge)
        g.add_node("tester", tester_agent)

        # Edges ---------------------------------------------------------------
        g.set_entry_point("researcher" if use_tavily else "coder")
        if use_tavily:
            g.add_edge("researcher", "coder")
        g.add_edge("coder",   "senior")
        g.add_edge("senior",  "reviewers")
        g.add_edge("reviewers", "merge")

        def after_merge(state: DevState):
            # break out if we've been rejected 3 times
            if state["reject_count"] >= 3:
                return END
            return "coder" if state["verdict"] == "REJECT" else "tester"

        g.add_conditional_edges("merge", after_merge)
        g.add_edge("tester", END)

        graph_app = g.compile()

        # 6️⃣  Execute ---------------------------------------------------------
        init: DevState = {
            "messages":     [base_system_block, f"USER: {query}"],
            "review_results": {},
            "verdict":      "REJECT",
            "reject_count": 0,
        }
        final = await graph_app.ainvoke(init)
        combined = "\n".join(GitHubMCPClient._flatten_messages(final["messages"]))
        #combined = "\n".join(_flatten_messages(final["messages"]))
        return self._extract_code_and_comments(combined)

    # ‑‑‑ interactive loop ----------------------------------------------------
    async def chat_loop(self) -> None:
        print("\nGitHub MCP Client Started!\nType your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                code, prose = await self.process_query(query)
                print("\n—‑‑ CODE \n" + (code or "[no code block]"))
                print("\n—‑‑ COMMENTS \n" + (prose or "[no prose]"))
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()

# ── script entry -------------------------------------------------------------
async def main() -> None:
    client = GitHubMCPClient()
    try:
        await client.connect_to_github_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
