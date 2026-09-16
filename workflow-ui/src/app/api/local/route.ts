import { NextResponse } from "next/server";
import { isLlmReady, readLocalResult, readOverrides } from "@/lib/local";

export const runtime = "nodejs";

export async function GET() {
  const [wynik, overrides, llmGotowy] = await Promise.all([
    readLocalResult(),
    readOverrides(),
    isLlmReady(),
  ]);
  return NextResponse.json({ wynik, overrides, llmGotowy });
}
