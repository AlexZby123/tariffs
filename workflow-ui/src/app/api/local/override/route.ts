import { NextResponse } from "next/server";
import {
  readOverrides,
  saveOverridesAndRetrain,
  type LocalRunOptions,
} from "@/lib/local";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json(await readOverrides());
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      przypisania: Record<string, string>;
      doucz: boolean;
      runOptions: LocalRunOptions;
    };
    const wynik = await saveOverridesAndRetrain(
      body.przypisania,
      body.doucz,
      body.runOptions,
    );
    return NextResponse.json({ ...wynik, overrides: await readOverrides() });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 400 });
  }
}
