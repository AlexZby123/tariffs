import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import { agenticDir } from "@/lib/workflow";

export const runtime = "nodejs";

const KATALOG_WEJSCIOWY = path.join(agenticDir, "dane_wejsciowe");
const DOZWOLONE = [".csv", ".xlsx", ".xls", ".xlsm", ".txt"];
const MAX_BAJTOW = 200 * 1024 * 1024;

/** Nazwa pliku bez katalogow i znakow, ktore moglyby wyjsc poza katalog docelowy. */
function bezpiecznaNazwa(nazwa: string): string {
  const sama = path.basename(nazwa).replace(/[^\w.\- ]+/g, "_");
  return sama.slice(-120) || "wejscie.csv";
}

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const plik = form.get("file");
    if (!(plik instanceof File)) {
      return NextResponse.json({ error: "No file in the request." }, { status: 400 });
    }
    if (plik.size > MAX_BAJTOW) {
      return NextResponse.json(
        { error: `File is larger than ${MAX_BAJTOW / 1024 / 1024} MB.` },
        { status: 413 },
      );
    }
    const nazwa = bezpiecznaNazwa(plik.name);
    if (!DOZWOLONE.includes(path.extname(nazwa).toLowerCase())) {
      return NextResponse.json(
        { error: `Unsupported file type. Allowed: ${DOZWOLONE.join(", ")}` },
        { status: 415 },
      );
    }

    await fs.mkdir(KATALOG_WEJSCIOWY, { recursive: true });
    const cel = path.join(KATALOG_WEJSCIOWY, nazwa);
    await fs.writeFile(cel, Buffer.from(await plik.arrayBuffer()));
    return NextResponse.json({ path: cel, name: nazwa, size: plik.size });
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
