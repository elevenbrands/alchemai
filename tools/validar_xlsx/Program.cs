using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;
using DocumentFormat.OpenXml.Validation;

int total = 0;
foreach (var path in args)
{
    using var doc = SpreadsheetDocument.Open(path, false);
    foreach (var version in new[] { FileFormatVersions.Office2010, FileFormatVersions.Office2019, FileFormatVersions.Microsoft365 })
    {
        var errors = new OpenXmlValidator(version).Validate(doc).ToList();
        total += errors.Count;
        Console.WriteLine($"{Path.GetFileName(path)} | {version}: {errors.Count} errores");
        foreach (var e in errors.Take(8))
            Console.WriteLine($"   {e.ErrorType} | {e.Part?.Uri} | {e.Path?.XPath} | {e.Description}");
    }
    int lists = 0, hidden = 0;
    foreach (var ws in doc.WorkbookPart!.WorksheetParts)
        foreach (var dv in ws.Worksheet.Descendants<DataValidation>())
        {
            lists++;
            if (dv.ShowDropDown?.Value == true) hidden++;   // in this format showDropDown=true HIDES the in-cell arrow
        }
    Console.WriteLine($"   listas de validación: {lists}; con la flecha oculta: {hidden}; hojas: {doc.WorkbookPart.Workbook.Sheets!.Count()}");
}
Console.WriteLine($"TOTAL de errores de esquema: {total}");
return total == 0 ? 0 : 1;
