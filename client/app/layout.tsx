import "./globals.css";
import Image from "next/image";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex flex-col">
        <main className="flex-grow">{children}</main>
        <footer className="w-full pb-4 px-6 border-t border-slate-200/30 mt-[-50px]">
          <div className="flex items-center justify-center gap-2">
            <span className="text-sm text-slate-600">Powered by</span>
            <Image
              src="/jojen-logo.webp"
              alt="Jojen"
              width={80}
              height={30}
              className="object-contain"
            />
          </div>
        </footer>
      </body>
    </html>
  );
}
